from transformers import pipeline


def grammar_checker():
    classifier = pipeline(task='text-classification', model='abdulmatinomotoso/English_Grammar_Checker')
    output = classifier("I will dog walk")
    print(output)


# Question Natural Language Inference (QNLI) dertermines if a piece of text contais enough information to answer a
# posed questions. This requires a model to perform logical reasoning
def qnli():
    classifier = pipeline(task='text-classification', model='cross-encoder/qnli-electra-base')
    # output = classifier("What is garbage collections?, Http head of the bloc is a performance overhead")
    output = classifier("What is purpose of garbage collections?, Garbage Collections help in Automatic memory "
                        "management")
    print(output)


def zero_shot_classification():
    classifier = pipeline(task='zero-shot-classification', model='facebook/bart-large-mnli')
    candidate_labels = ['maths', 'science', 'economics', 'sports', 'none']
    output = classifier("sum of two positive number is always greater than individual numbers", candidate_labels)
    print(output)


def abstractive_summary():
    with open('QA.txt', 'r') as file:
        text = file.read()
    chunk_size = len(text) // 4
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    for i in chunks:
        print(i)

    summarizer = pipeline(task='summarization', model="cnicu/t5-small-booksum", max_length=100, min_length=50)
    output = summarizer(chunks,truncation=True)
    print(output)


if __name__ == "__main__":
    abstractive_summary()
