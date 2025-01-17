from datasets import Audio,load_dataset ,load_dataset_builder

songs = load_dataset('DynamicSuperb/MusicGenreClassification_FMA',split='test')
print(songs[0])

print( "Using index")
for i in range(5):
    print(songs[i])

print("Using for each")
for song in list(songs[:5]):
    print(type(song))  # Confirm if song is a dictionary
    print(song)
