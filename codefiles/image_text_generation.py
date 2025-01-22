from transformers import pipeline, AutoProcessor, AutoModelForCausalLM
from PIL import Image


def using_pipeline_directly():
    pipe = pipeline(task='image-to-text', model='microsoft/git-base-coco')
    output = pipe('../test-images/test.png')
    print(output)


def using_auto_classes():
    processor = AutoProcessor.from_pretrained("microsoft/git-base-coco")
    model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco")
    model = model.to('xpu')
    img = Image.open('../test-images/test.png')
    pixel = processor(img, return_tensors="pt").to('xpu')
    pixel_values = pixel.pixel_values
    output = model.generate(pixel_values=pixel_values, max_length=1000)
    generated_text = processor.batch_decode(output, skip_special_tokens=True)
    print(generated_text[0])


if __name__ == "__main__":
    using_auto_classes()
