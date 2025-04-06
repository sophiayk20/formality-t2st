### For each audio file, extract embeddings for each segment
### Each audio file's speaker embedding is averaged across the segments
### Clustering is performed with num clusters = 2 for each gender with KMeans
### Speaker embeddings have higher dimensionality, so we reduce dimension to 2D with t-SNE

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torchaudio
from pyannote.audio import Inference, Model, Pipeline
from pyannote.core import Segment
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

HUGGINGFACE_TOKEN="" # Add huggingface token here
AUDIO_FILES_DIRECTORY="/content/drive/MyDrive/fr-informal-random/"
BASE_FOLDER="/content/drive/MyDrive"
FORMALITY="informal"

# init speaker diarization pipeline and embedding model
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=HUGGINGFACE_TOKEN
)
# speaker embedding model
embedding_model = Model.from_pretrained("pyannote/embedding", use_auth_token=HUGGINGFACE_TOKEN)

# path to audio files
audio_files_directory = AUDIO_FILES_DIRECTORY
audio_files = [f for f in os.listdir(audio_files_directory) if f.endswith('.wav')]  # Modify extension if needed
audio_files = [f for f in audio_files if f.startswith('male') or f.startswith('female')]

# dictionary to store one embedding for each file
file_embeddings = []
file_names = []
gender_dict = {}
# error: kernel size 5 in embedding model architecture, for use with pyannote.audio
min_segment_duration = 0.5
inference = Inference(embedding_model, window="whole")

# move inference and pipeline to CUDA
inference.to(torch.device("cuda"))
pipeline.to(torch.device("cuda"))

# process each audio file in the directory
for audio_file_name in tqdm(audio_files):
    # extract gender from the filename (e.g., 'male-1.wav' -> 'male')
    gender = audio_file_name.split('-')[0]
    audio_file_path = os.path.join(audio_files_directory, audio_file_name)
    diarization = pipeline(audio_files_directory+audio_file_name)
    waveform, sample_rate = torchaudio.load(audio_files_directory+audio_file_name)
    # duration of audio in seconds
    audio_duration = waveform.size(1) / sample_rate
    embeddings= []
    num_segments_processed = 0

    # For each segment, extract embeddings
    for segment, _, speaker in diarization.itertracks(yield_label=True):
      # allow the segment to be large enough for pipeline processing (segment has start and end times)
      #[ 00:00:01.279 -->  00:00:04.148]: 2 seconds
      #[ 00:00:04.924 -->  00:00:09.126]
      #[ 00:00:10.139 -->  00:00:13.125]
      if min(segment.end, audio_duration) - segment.start < min_segment_duration:
          continue

      # prevent model from oversegmenting past audio duration
      segment_end = min(segment.end, audio_duration)
      excerpt = Segment(segment.start, segment_end)
      embedding = inference.crop(audio_file_path, excerpt)
      num_segments_processed += 1

      # append segment embedding
      embeddings.append(embedding)

    # if at least 1 embedding was added and not skipped over for short segments
    if embeddings:
      aggregated_embedding = np.mean(embeddings, axis=0)

      # Store the aggregated embedding and file name
      file_embeddings.append(aggregated_embedding)
      file_names.append(audio_file_name)
      gender_dict[audio_file_name] = gender

# convert list of embeddings to a numpy array
embedding_matrix = np.vstack(file_embeddings)

gender_labels = [gender_dict[filename] for filename in file_names]
male_indices = [i for i, label in enumerate(gender_labels) if label == "male"]
female_indices = [i for i, label in enumerate(gender_labels) if label == "female"]

# perform clustering with 2 speakers each for each gender
kmeans_male = KMeans(n_clusters=2, random_state=42)
kmeans_female = KMeans(n_clusters=2, random_state=42)

# cluster embeddings first on gender
kmeans_male_labels = kmeans_male.fit_predict(embedding_matrix[male_indices])
kmeans_female_labels = kmeans_female.fit_predict(embedding_matrix[female_indices])

# assign clusters to labels
cluster_labels = np.array([None] * len(embedding_matrix))
cluster_labels[male_indices] = kmeans_male_labels
cluster_labels[female_indices] = kmeans_female_labels

# map gender and speaker information
speaker_labels = [f"{gender[0].upper() + gender[1:]} {cluster_labels[i]}" for i, gender in enumerate(gender_labels)]
gender_labels_map = [gender_dict[filename] for filename in file_names]

# save relevant files to numpy
np.save(f'{BASE_FOLDER}/{FORMALITY}_speaker_embeddings.npy', file_embeddings)  # Save the embeddings
np.save(f'{BASE_FOLDER}/{FORMALITY}_speaker_labels.npy', speaker_labels)        # Save the corresponding labels (Male 1, Female 0)
np.save(f'{BASE_FOLDER}/{FORMALITY}_gender_labels.npy', gender_labels)      # Save the gender information
np.save(f'{BASE_FOLDER}/{FORMALITY}_filenames.npy', file_names)

# t-SNE to reduce the dimensionality of the embeddings
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
embeddings_2d = tsne.fit_transform(embedding_matrix)

# plot t-SNE visualization with different color for gender and marker style for speaker ID
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=embeddings_2d[:, 0], y=embeddings_2d[:, 1],
    hue=speaker_labels,  # color by speaker cluster
    style=speaker_labels,  # marker style by gender
    palette="Set2",  # color palette for speakers
    markers=["o", "s"],  # circle for male, square for female
    s=100  # size of the markers
)
plt.title(f"{FORMALITY[0].upper() + FORMALITY[1:].lower()} Speaker Clustering with t-SNE (Colored by Speaker, Styled by Gender)")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.legend(title="Speaker", loc="upper right")
plt.show()
