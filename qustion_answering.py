from transformers import pipeline
import os


def visual_answer_question(question, context):
    question_answering = pipeline(task='visual-question-answering', model='dandelin/vilt-b32-finetuned-vqa')
    output = question_answering(question=question, image=context)
    for i in output:
        print(i[0]['answer'])


if __name__ == "__main__":
    visual_answer_question(["What is the color of the fruit?","what is this fruit"], "test-images/IMG_TEST_1.jpg")
