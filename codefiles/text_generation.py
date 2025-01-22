from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch


def using_pipeline_directly():
    # Use the pipeline for text generation
    pipe = pipeline("text-generation", model="openai-community/gpt2-large")
    output = pipe("Write the summary of jayshankar prashad poem ,Aatmakathya. /n", max_length=500, num_return_sequences=1, truncation=True)
    print(output[0]['generated_text'])


def using_auto_classes():
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2-large")
    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2-large")

    # Prepare the input prompt
    prompt = "Explain G1 GC in java. How it works?"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate text with parameters adjusted for longer output
    generated_text = model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=1000,  # Increase the maximum length of generated text
        do_sample=True,  # Enable sampling for more diverse output
        temperature=0.7,  # Control randomness; higher values are more creative
        top_k=50,        # Keep only the top k tokens for each step
        top_p=0.95,      # Use nucleus sampling with this probability
        num_return_sequences=1,  # Number of sequences to return
        no_repeat_ngram_size=2  # Avoid repeating ngrams
    )

    # Decode and print the generated text
    output = tokenizer.decode(generated_text[0], skip_special_tokens=True)
    print(output)


if __name__ == "__main__":
    using_pipeline_directly()

