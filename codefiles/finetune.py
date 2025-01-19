from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, pipeline
from datasets import load_dataset
import torch


def load_and_preprocess_dataset(dataset_name, train_range, test_range, tokenizer):
	"""Load and preprocess the dataset."""
	dataset = load_dataset(dataset_name, split="train")
	dataset_train = dataset.select(train_range)
	dataset_test = dataset.select(test_range)
	dataset_train = dataset_train.map(lambda row: tokenizer(row["text"], truncation=True, padding="max_length",return_tensors='pt'),
	                      keep_in_memory=True,batched=True)
	return dataset_train, dataset_test


def create_trainer(model, train_dataset, eval_dataset, output_dir):
	"""Initialize the Trainer."""
	training_args = TrainingArguments(
		output_dir=output_dir
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
	model_name = "bert-base-cased"


	model = AutoModelForSequenceClassification.from_pretrained(model_name)
	if torch.cuda.is_available():
		print(torch.cuda.is_available())
		model = torch.compile(model, backend="amd")

	# Move model to GPU
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to('cuda')
	tokenizer = AutoTokenizer.from_pretrained(model_name)

	# Toggle this block if dataset preprocessing is needed
	preprocess_data = True
	if preprocess_data:
		print("Loading and preprocessing the dataset...")
		train_range = range(1, 10)
		test_range = range(10, 20)
		dataset_name = "stanfordnlp/imdb"
		dataset_train, dataset_test = load_and_preprocess_dataset(dataset_name, train_range, test_range, tokenizer)
		print("Dataset preprocessing complete.")

	# Toggle this block if training is needed
	run_training = True
	if run_training:
		print("Starting model training...")
		output_dir = "./results"
		trainer = create_trainer(model, dataset_train, dataset_test, output_dir)
		trainer.train()
		print("Model training complete.")

	# Always run evaluation
	print("Evaluating the model...")
	fine_tuned_model_path = "./fine_tuned_model"

	if run_training:  # Save only if training was run
		trainer.save_model(fine_tuned_model_path)
		tokenizer.save_pretrained(fine_tuned_model_path)

	sample_review = """
    This is just a precious little diamond. The play, the script are excellent. I cant compare this movie with anything else, 
    maybe except the movie "Leon" wonderfully played by Jean Reno and Natalie Portman. But... What can I say about this one? 
    This is the best movie Anne Parillaud has ever played in (See please "Frankie Starlight", she's speaking English there) 
    to see what I mean. The story of young punk girl Nikita, taken into the depraved world of the secret government forces 
    has been exceptionally over used by Americans. Never mind the "Point of no return" and especially the "La femme Nikita" TV series. 
    They cannot compare the original believe me! Trash these videos. Buy this one, do not rent it, BUY it. BTW beware of the subtitles 
    of the LA company which "translate" the US release. What a disgrace! If you cant understand French, get a dubbed version. 
    But you'll regret later :)
    """
	classifier = pipeline("sentiment-analysis", model=fine_tuned_model_path)
	print(classifier(sample_review))
