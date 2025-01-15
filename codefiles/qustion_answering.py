from transformers import pipeline
import os

question_answering = pipeline(task='visual-question-answering', model='dandelin/vilt-b32-finetuned-vqa')
dqa= pipeline(task='document-question-answering',model ='impira/layoutlm-document-qac')
def visual_answer_question(question, context):

    output = question_answering(question=question, image=context)
    for i in output:
        print(i[0]['answer'])

def doc_question_answer(question, context):
    results = dqa(image= context, question=question)
    print(results)



if __name__ == "__main__":
    all_files = os.listdir('../test-pdfs')

    for file in all_files:
        #visual_answer_question(["What is the color of the fruit?","what is this fruit"], "../test-images/"+file)
        doc_question_answer('What is this document?',context="../test-pdfs/"+file)
