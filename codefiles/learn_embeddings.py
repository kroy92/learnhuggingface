import numpy as np
from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset


def test_simple_embeddings():
    embedder = SentenceTransformer('all-MiniLM-L6-v2', device='xpu')
    embeddings = embedder.encode(["This is a test sentence."])
    print(embeddings)


def create_embeddings_for_dataset(dataset):
    embedder = SentenceTransformer('all-MiniLM-L6-v2', device='xpu')
    embeddings = embedder.encode(dataset['text'])
    return embeddings


def create_embeddings(lst):
    embedder = SentenceTransformer('all-MiniLM-L6-v2', device='xpu')
    embeddings = embedder.encode(lst)
    return embeddings


def load_external_dataset():
    dataset = load_dataset('sentence-transformers/simple-wiki', split='train')
    return dataset


def save_embeddings(embeddings, filename='embeddings.npy'):
    np.save(filename, embeddings)


def load_embeddings(filename='embeddings.npy'):
    return np.load(filename)


if __name__ == "__main__":
    dst = load_external_dataset()

    # Create embeddings if not already saved
    try:
        embeds = load_embeddings()
        print("Embeddings loaded from disk.")
    except FileNotFoundError:
        print("Embeddings not found on disk, creating new ones.")
        embeds = create_embeddings(dst)
        save_embeddings(embeddings=embeds)

    # Here you can use 'embeds' for further processing or analysis
    print(embeds.shape)  # Just to show that we've got something

    query = ('Nuneaton is well-known for the novelist George Eliot, who was born on a farm near the town in 1819 and '
             'spent her early years living there.')
    query_embedding = create_embeddings([query])
    hits = util.semantic_search(query_embedding, embeds, top_k=2)
    print(hits[0])
    for hit in hits[0]:
        print(dst['text'][hit['corpus_id']], hit['score'])
