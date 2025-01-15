from transformers import image_transforms, pipeline
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os


def transform_image():

    files= os.listdir('../test-images')
    all_cis = []
    for file in files:
        original_image = Image.open(f'../test-images/{file}')
        image_array = np.array(original_image)
        #ci = image_transforms.center_crop(image=image_array, size=(800, 800))
        all_cis.append(image_array)
    return all_cis


def classify_image(ci):
    image_classifier = pipeline(task='image-classification', model= 'jazzmacedo/fruits-and-vegetables-detector-36')
    output = image_classifier([Image.fromarray(i) for i in ci])
    for i in output:
        print(i[0]['label'])


if __name__ == "__main__":
    cropped_images = transform_image()
    classify_image(cropped_images)
