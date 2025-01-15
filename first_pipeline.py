from transformers import pipeline


def main():
    # Use the pipeline for sentiment analysis
    pipe = pipeline("sentiment-analysis", model="siebert/sentiment-roberta-large-english", device=0)
    output = pipe(
        "I am absolutely thrilled and overjoyed! This experience has been incredibly rewarding and I am filled with "
        "immense gratitude and appreciations")
    print(output)


if __name__ == "__main__":
    main()
