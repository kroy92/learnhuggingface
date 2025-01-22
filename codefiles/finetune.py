from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, pipeline
from datasets import load_dataset, concatenate_datasets
import torch
from torch.nn.parallel import DataParallel


def load_and_preprocess_dataset(dataset_name, train_range, test_range, tokenizer):
    """Load and preprocess the dataset."""
    dataset = load_dataset(dataset_name, split="train")
    dataset_pas = dataset.filter(lambda row: row["label"] == 1)
    dataset_neg = dataset.filter(lambda row: row["label"] == 0)
    dataset_train_pass = dataset_pas.select(train_range)
    dataset_train_neg = dataset_neg.select(train_range)
    dataset_train = concatenate_datasets([dataset_train_pass, dataset_train_neg])
    dataset_test_pass = dataset_pas.select(test_range)
    dataset_test_neg = dataset_neg.select(test_range)
    dataset_test = concatenate_datasets([dataset_test_pass, dataset_test_neg])
    dataset_train = dataset_train.map(
        #lambda row: tokenizer(row["text"], truncation=True, padding="max_length", return_tensors='pt',),
        lambda row: tokenizer(row["text"], truncation=True, padding=True, max_length=512, return_tensors='pt'),
        keep_in_memory=True, batched=True)
    return dataset_train, dataset_test


def create_trainer(model, train_dataset, eval_dataset, output_dir):
    """Initialize the Trainer."""
    training_args = TrainingArguments(
        output_dir=output_dir,
        use_cpu=False
    )
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )


def save_and_evaluate_model(trainer, save_path, sample_text):
    """Save the fine-tuned model and evaluate it with a sample text."""
    trainer.save_model(save_path)
    classifier = pipeline("sentiment-analysis", model=save_path)
    print("Sample Sentiment Analysis Result:")
    print(classifier(sample_text))


if __name__ == "__main__":
    # Define model and tokenizer
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"

    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    if torch.cuda.is_available():
        print(torch.cuda.is_available())
        # model = torch.compile(model, backend="amd")
    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Toggle this block if dataset preprocessing is needed
    preprocess_data = False
    if preprocess_data:
        print("Loading and preprocessing the dataset...")
        train_range = range(1, 500)
        test_range = range(501, 1000)
        dataset_name = "stanfordnlp/imdb"
        dataset_train, dataset_test = load_and_preprocess_dataset(dataset_name, train_range, test_range, tokenizer)
        print("Dataset preprocessing complete.")

    # Toggle this block if training is needed
    run_training = False
    if run_training:
        print("Starting model training...")
        output_dir = "./results"
        trainer = create_trainer(model, dataset_train, dataset_test, output_dir)
        trainer.train()
        print("Model training complete.")

    # Always run evaluation
    print("Evaluating the model...")
    fine_tuned_model_path = "./fine_tuned_model1"

    if run_training:  # Save only if training was run
        trainer.save_model(fine_tuned_model_path)
        tokenizer.save_pretrained(fine_tuned_model_path)

    sample_review = '''Kangana Ranaut’s Emergency delivers a powerful attempt to depict one of India’s most 
    controversial political periods, with her portrayal of Indira Gandhi commanding attention through sheer intensity 
    and presence. However, the film’s narrative struggles with consistency, as uneven pacing and tonal shifts disrupt 
    its flow. While some scenes are visually striking and effectively capture the drama of the era, others feel 
    rushed or underdeveloped, limiting the emotional depth and complexity of the characters. Despite its 
    shortcomings, while imperfect, is engaging for those 
    interested in political storytelling.Emergency remains a noteworthy cinematic effort. good movie'''
    classifier = pipeline("text-classification", model=fine_tuned_model_path, truncation=True)
    print(classifier(sample_review))
    new_data= load_dataset('stanfordnlp/imdb',split='unsupervised')

    new_data= [i['text'] for i in new_data]
    result = classifier(new_data)
    result_list= [i['label'] for i in result]
    print('Negative Reviews:',result_list.count('NEGATIVE'))
    print('Positive Reviews:',result_list.count('POSITIVE'))
