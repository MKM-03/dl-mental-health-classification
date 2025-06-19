import json
import numpy as np
from gensim.models import FastText
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained custom FastText model to extract word embeddings
ft = FastText.load("models/FT models/fasttext_large5.model")
embedding_size = ft.vector_size

# Load the preprocessed texts which are tokenized sequences of texts (a list of lists)
with open('tokenized/large_tokenized_2.txt', 'r') as f:
    tokenized_texts = json.loads(f.read())


embeddings = []
for text in tokenized_texts:
    # Get the embedding for each token, or zero if the token is not in the model's vocabulary
    text_embeddings = [ft.wv[token] if token in ft.wv else np.zeros(embedding_size) for token in text]
    embeddings.append(text_embeddings)

# Pad sequences to ensure consistent sequence length of 100
max_sequence_length = 100

# Pad sequences to ensure all texts have the same length
padded_embeddings = pad_sequences(embeddings, padding='post', maxlen=max_sequence_length, dtype='float32')

# The shape of padded_embeddings will be [batch_size, 100, embedding_size]
# For us it will be (100, 100) since batch size would be specified later
print(f"Padded embeddings shape: {padded_embeddings.shape}")
# Save as a .npy (NumPy array) file
np.save('embeddings_dir/padded_embeddings_large_100.npy', padded_embeddings)
