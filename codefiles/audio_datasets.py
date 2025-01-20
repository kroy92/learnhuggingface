import librosa
from datasets import load_dataset , Audio
from transformers import pipeline


def load_and_preprocess_dataset(dataset_name, split='test', min_duration=30):
    """Load the dataset and filter songs by duration."""
    print("Loading dataset...")
    songs = load_dataset(dataset_name, split=split)
    print("Dataset loaded. Example:", songs[0])

    # Calculate durations
    print("Calculating durations...")
    durations = [
        librosa.get_duration(y=row['array'], sr=row['sampling_rate'])
        for row in songs['audio']
    ]

    # Add duration column
    print("Adding duration column...")
    songs = songs.add_column("duration", durations)
    # songs = songs.cast_column

    songs = songs.cast_column("audio",Audio(sampling_rate=16000))
    print("Dataset Changed. Example:", songs[0])

    # Filter songs by duration
    print("Filtering songs longer than", min_duration, "seconds...")
    filtered_songs = songs.filter(lambda row: row['duration'] > min_duration)

    print(f"Filtered dataset contains {len(filtered_songs)} songs.")
    return filtered_songs


def classify_songs(filtered_songs, classifier_pipeline):
    """Classify songs and print results."""
    print("Classifying songs...")
    for song in filtered_songs:
        audio = song['audio']['array']
        prediction = classifier_pipeline(audio, top_k=1)
        print(f"Path: {song['audio']['path']}, Prediction: {prediction}, Label: {song['label']}")


def main():
    dataset_name = 'DynamicSuperb/MusicGenreClassification_FMA'
    classifier_model = 'SeyedAli/Musical-genres-Classification-Hubert-V1'

    # Load and preprocess the dataset
    filtered_songs = load_and_preprocess_dataset(dataset_name)

    # Initialize the audio classification pipeline
    print("Initializing classifier...")
    classifier = pipeline(task='audio-classification', model=classifier_model)

    # Classify and print results
    classify_songs(filtered_songs, classifier)


if __name__ == "__main__":
    main()
