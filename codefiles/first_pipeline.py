from transformers import pipeline, AutoModelForSequenceClassification , AutoTokenizer

def using_pipeline_directly():
    # Use the pipeline for sentiment analysis
    pipe = pipeline("sentiment-analysis", model="siebert/sentiment-roberta-large-english")
    output = pipe("I absolutely happy everything about this situationI am absolutely thrilled and overjoyed! This experience has been incredibly rewarding and I am filled with immense gratitude and appreciations")
    print(output)

def using_auto_classes():
    # Download the model and the tokenizer
    print ("Using Auto classes")
    model = AutoModelForSequenceClassification.from_pretrained("siebert/sentiment-roberta-large-english")
    tokenizer = AutoTokenizer.from_pretrained("siebert/sentiment-roberta-large-english")

    #Create the Pipeline

    sentimentAnalysis = pipeline(task = 'sentiment-analysis' ,model = model , tokenizer= tokenizer )

    # use the pipeline
    output = sentimentAnalysis(["Life is so beautiful and happy","Student life is so depressing"])

    print(output)



if __name__ == "__main__":
    # using_pipeline_directly()
    using_auto_classes()

