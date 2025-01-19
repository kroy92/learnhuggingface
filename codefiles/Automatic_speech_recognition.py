from datasets import load_dataset , Audio , load_dataset_builder
from transformers import pipeline
from evaluate import load

ldb = load_dataset("alibabasglab/LJSpeech-1.1-48kHz",split='train')
ldb = ldb.cast_column('audio', Audio(sampling_rate=16000))
print(len(ldb))
print(ldb[0])
def data():
	for i in ldb:
		yield i['audio']['array']




print(ldb[0])

transcriber_facebook = pipeline(task= 'automatic-speech-recognition', model = 'facebook/wav2vec2-base-960h')
transcriber_ms = pipeline(task = 'automatic-speech-recognition', model = 'microsoft/speecht5_asr')
transcriber_whisper = pipeline(task = 'automatic-speech-recognition', model = 'openai/whisper-small')

true_sentence=transcriber_whisper([i['audio']['array'] for i in ldb])
true_sentence=[i['text'].lower() for i in true_sentence]

facebook_prediction = transcriber_facebook([i['audio']['array'] for i in ldb])
facebook_prediction= [i['text'].lower() for i in facebook_prediction]

microsoft_prediction = transcriber_ms([i['audio']['array'] for i in ldb])
microsoft_prediction= [i['text'].lower() for i in microsoft_prediction]

wer= load('wer')
wer_score_facebook = wer.compute(predictions=facebook_prediction, references =true_sentence)
wer_score_microsoft = wer.compute(predictions = microsoft_prediction, references = true_sentence)

print(wer_score_facebook,wer_score_microsoft)

for i,j,k in zip(microsoft_prediction,facebook_prediction,true_sentence):
	print(f'''
1.{i}
2. {j}
3. {k}''')
	print('*' * 30)








