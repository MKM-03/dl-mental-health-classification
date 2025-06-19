import re
import string
import nltk
import json
import contractions
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from multiprocessing import Pool
from tqdm import tqdm

# Download punkt, stopwords and wordnet for preprocessing
# or load them if already downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load global resources
stop_words = set(stopwords.words('english'))

# Define regex patterns
url_pattern = re.compile(r'http\S+')
html_tag_pattern = re.compile(r"<[^>]+>")
hashtag_pattern = re.compile(r'#\w+')
subreddit_pattern = re.compile(r'r/\w+')

# Regex pattern to match words that repeat twice or more consecutively
repeated_pattern = re.compile(r'\b([a-zA-Z]+)\1+\b')


# Text cleaning and preprocessing function
def preprocess_text(text):

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

    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]

    # Remove short tokens
    tokens = [token for token in tokens if len(token) > 2]

    return tokens


# This is used for parallel processing to speed up the process
if __name__ == "__main__":
    # Load the CSV files containing raw texts into a pandas DataFrame
    # Remove the generic mentaillness class as it may add noise
    # Remove duplicates and NaN from whole dataset
    df = pd.read_csv('data/cleaned_reddit.csv', encoding='utf-8')
    df = df[df.label != 'mentalillness']
    df = df.drop_duplicates().dropna()
    print(df.label.value_counts())
    texts = df.text.values

    # Process texts in parallel with multiprocessing
    with Pool() as pool:
        preprocessed_texts = list(tqdm(pool.imap(preprocess_text, texts), total=len(texts), desc="Preprocessing Texts"))

    # Save the preprocessed texts to a file
    with open('tokenized/large_tokenized_2.txt', 'w') as f:
        f.write(json.dumps(preprocessed_texts))
