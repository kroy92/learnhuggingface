from huggingface_hub import HfApi

api = HfApi()

# Limit the number of models and fetch minimal metadata
models = list(api.list_models(limit=5, fetch_config=False))

for i in models:
    print(f"Model ID: {i.modelId}, Tags: {i.tags}")
