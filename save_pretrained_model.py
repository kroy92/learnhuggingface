from transformers import AutoModel
model = AutoModel.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
model.save_pretrained(save_directory=f'model/distilbert-base-uncased-finetuned-sst-2-english')
print("Model saved successfully.")