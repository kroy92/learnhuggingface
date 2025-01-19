from datasets import load_dataset, Audio
from transformers import pipeline
from evaluate import load


def load_and_preprocess_data(dataset_name, split, sampling_rate=16000):
	"""Load and preprocess the dataset."""
	dataset = load_dataset(dataset_name, split=split)
	dataset = dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))
	print(f"Loaded dataset with {len(dataset)} samples.")
	return dataset


def transcribe_audio(transcriber, audio_data):
	"""Transcribe audio using the specified pipeline."""
	return [result["text"].lower() for result in transcriber(audio_data)]


def calculate_wer(predictions, references):
	"""Calculate Word Error Rate (WER)."""
	wer = load("wer")
	return wer.compute(predictions=predictions, references=references)


def batch_process(dataset, batch_size=8):
	"""Yield batches of audio arrays from the dataset."""
	for i in range(0, len(dataset), batch_size):
		batch = dataset[i: i + batch_size]["audio"]
		yield [sample["array"] for sample in batch]


# Load dataset
dataset_name = "alibabasglab/LJSpeech-1.1-48kHz"
dataset = load_and_preprocess_data(dataset_name, split="train")

# Initialize transcribers
transcriber_facebook = pipeline(task="automatic-speech-recognition", model="facebook/wav2vec2-base-960h")
transcriber_ms = pipeline(task="automatic-speech-recognition", model="microsoft/speecht5_asr")
transcriber_whisper = pipeline(task="automatic-speech-recognition", model="openai/whisper-small")

# Batch processing
batch_size = 8
facebook_predictions = []
microsoft_predictions = []
true_sentences = []

for audio_batch in batch_process(dataset, batch_size=batch_size):
	# Transcribe each batch
	whisper_transcriptions = transcribe_audio(transcriber_whisper, audio_batch)
	facebook_transcriptions = transcribe_audio(transcriber_facebook, audio_batch)
	microsoft_transcriptions = transcribe_audio(transcriber_ms, audio_batch)

	# Collect results
	true_sentences.extend(whisper_transcriptions)
	facebook_predictions.extend(facebook_transcriptions)
	microsoft_predictions.extend(microsoft_transcriptions)

# Compute WER scores
wer_score_facebook = calculate_wer(facebook_predictions, true_sentences)
wer_score_microsoft = calculate_wer(microsoft_predictions, true_sentences)

print(f"WER Score (Facebook): {wer_score_facebook}")
print(f"WER Score (Microsoft): {wer_score_microsoft}")

# Print results for inspection
for i, (true_text, fb_text, ms_text) in enumerate(zip(true_sentences, facebook_predictions, microsoft_predictions),
                                                  start=1):
	print(f"""
Sample {i}:
1. Microsoft: {ms_text}
2. Facebook: {fb_text}
3. True Sentence: {true_text}
{'*' * 30}
""")
