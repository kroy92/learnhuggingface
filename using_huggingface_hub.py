from huggingface_hub import HfApi
api = HfApi()

# Limit the number of models and fetch minimal metadata
# models = list(api.list_models(limit=5))

# Filter using task , skill, sort by downloads

models = api.list_models(task='text-classification',sort='downloads', direction=-1, limit=1)

print(list(models))
