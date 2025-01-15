"""
Key components

1. Normalisation : cleaning the text and applying transformation -such as casing
Removing whitespaces , Accents , Lowercasing , removing punctuation sometimes

2. Pre-tokenization : splitting into smaller tokens
Tokenization model
"""
from transformers import AutoTokenizer, GPT2Tokenizer
def using_auto_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    print(tokenizer.backend_tokenizer.normalizer.normalize_str("Howdy,how are you"))
    ls=tokenizer.tokenize("Howdy,how are you")
    print(ls)

def using_gpt2_tokenizer():
	gpt_tokenizer= GPT2Tokenizer.from_pretrained("gpt2")
	print(gpt_tokenizer.tokenize("Howdy, how are you"))

if __name__ == '__main__':
	using_gpt2_tokenizer()

