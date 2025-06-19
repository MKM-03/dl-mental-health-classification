import re
import string
import contractions
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import FastText
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Download punkt, stopwords and wordnet for preprocessing
# or load them if already downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load global resources once
stop_words = set(stopwords.words('english'))


# Text cleaning and preprocessing function
def preprocess_text(text):
    # Define regex patterns
    url_pattern = re.compile(r'http\S+')
    html_tag_pattern = re.compile(r"<[^>]+>")
    hashtag_pattern = re.compile(r'#\w+')
    subreddit_pattern = re.compile(r'r/\w+')
    repeated_pattern = re.compile(r'\b([a-zA-Z]+)\1+\b')

    # Apply basic data preprocessing such as:
    # Removing URLs, HTML tags, hashtags, subreddit mention
    text = url_pattern.sub('', text)
    text = html_tag_pattern.sub('', text)
    text = hashtag_pattern.sub('', text)
    text = subreddit_pattern.sub('', text)

    # Expand contractions such as "haven't" to "have not" and misspelled ones such as 'Im" to 'I am'
    text = contractions.fix(text.lower())

    text = repeated_pattern.sub('', text)

    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)

    # Remove special characters and punctuation, keeping only letters and spaces
    text = re.sub(r'[^A-Za-z\s]', '', text)

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]

    # Remove short tokens
    tokens = [token for token in tokens if len(token) > 2]

    return tokens


def get_embeddings(tokenized):
    ft = FastText.load("models/FT models/fasttext_large5.model")

    embeddings = []
    for text in tokenized:
        # Get the embedding for each token, or zero if the token is not in the model's vocabulary
        text_embeddings = [ft.wv[token] if token in ft.wv else np.zeros(100) for token in text]
        embeddings.append(text_embeddings)

    # Pad sequences to ensure consistent sequence length of 100
    max_sequence_length = 100

    # Pad sequences to ensure all texts have the same length
    padded_embeddings = pad_sequences(embeddings, padding='post', maxlen=max_sequence_length, dtype='float32')

    return padded_embeddings


def make_prediction(input_list):
    preprocessed_input = [preprocess_text(text) for text in input_list]
    embedded_input = get_embeddings(preprocessed_input)

    model = tf.keras.models.load_model('models/checkpoint_large')

    prediction = model.predict(embedded_input)
    print(prediction)

    # Aggregate predictions
    average_prediction = np.mean(prediction, axis=0)  # Mean probability across inputs
    print("Average Prediction (Probabilities):", average_prediction)

    # Final class based on maximum probability
    final_class = np.argmax(average_prediction)
    print("Final Output (Weighted by Confidence):", final_class)


input_text = ["Yesterday was a shitshow. I was flying hi all day long, buzzing, erratic talking, being everybody’s everything. I was spinning by the end of the day and needed my meds + insomnia meds to even sleep 4 hours. Get up for work, and I can feel it coming on. All day long, I lost little pieces of myself and by noon -I was devoid of it all - energy, emotion, any sense of joy. I took a 1/2 FMLA day and left. Slept 4 hours and am just lying here with nothing left in me. I think I need a med rearrangement; my ocd and obsessive thoughts about death and people leaving me is unbearable and uncontrollable. Im not in any way going to hurt myself physically. Right now, I can barely constitute as human. I just wish I could get a ton of sleep and wake up new. But we know how BP1 depression goes. Just needed to vent. I find myself incredibly lucky with a good job, beautiful home, a dog and a cat (no children), and a supportive and understanding husband. I’m just unlucky with my brain chemistry. I hate how many days I lose each year to this fucking illness."]

# input_text = ['testing testing', 'test test']
make_prediction(input_text)
