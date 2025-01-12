from transformers import AutoModel
from datasets import load_dataset_builder,load_dataset

def save_model():
	model = AutoModel.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
	model.save_pretrained(save_directory=f'model/distilbert-base-uncased-finetuned-sst-2-english')
	print("Model saved successfully.")

def see_dataset():
	data_builder = load_dataset_builder('stanfordnlp/imdb')
	print(data_builder.info)
	print(data_builder.info.description)
	print(data_builder.info.dataset_size)
	print(data_builder.info.features)

	data = load_dataset('stanfordnlp/imdb',split='train')


def filter_dataset():
	data = load_dataset('stanfordnlp/imdb', split='train')
	filtered = data.filter(lambda row: row['label']==0)
	sliced = filtered.select([1,2])
	print(sliced[0]['text'])
	pass



filter_dataset()

