import tkinter as tk
from tkinter import messagebox
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
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load global resources once
stop_words = set(stopwords.words('english'))
ft = FastText.load("models/FT models/fasttext_large5.model")

model = tf.keras.models.load_model('models/checkpoint_large')
# Define regex patterns
url_pattern = re.compile(r'http\S+')
html_tag_pattern = re.compile(r"<[^>]+>")
hashtag_pattern = re.compile(r'#\w+')
subreddit_pattern = re.compile(r'r/\w+')
repeated_pattern = re.compile(r'\b([a-zA-Z]+)\1+\b')


# Text cleaning and preprocessing function
def preprocess_text(text):
    # Apply basic data preprocessing such as:
    text = url_pattern.sub('', text)
    text = html_tag_pattern.sub('', text)
    text = hashtag_pattern.sub('', text)
    text = subreddit_pattern.sub('', text)
    text = contractions.fix(text.lower())  # Expand contractions
    text = repeated_pattern.sub('', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)  # Remove special characters and punctuation

    tokens = word_tokenize(text)

    # Remove stopwords and short tokens
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [token for token in tokens if len(token) > 2]

    return tokens


def get_embeddings(tokenized):

    embeddings = []
    for text in tokenized:
        text_embeddings = [ft.wv[token] if token in ft.wv else np.zeros(100) for token in text]
        embeddings.append(text_embeddings)

    # Pad sequences to ensure consistent sequence length of 100
    max_sequence_length = 100
    padded_embeddings = pad_sequences(embeddings, padding='post', maxlen=max_sequence_length, dtype='float32')

    return padded_embeddings


def make_prediction(input_list):
    preprocessed_input = [preprocess_text(text) for text in input_list]
    embedded_input = get_embeddings(preprocessed_input)

    prediction = model.predict(embedded_input)
    predicted_classes = np.argmax(prediction, axis=1)

    predicted_class_name = ''
    # Map the predicted index (0-4) to the corresponding class name
    if predicted_classes[0] == 0:
        predicted_class_name = "Anxiety"
    elif predicted_classes[0] == 1:
        predicted_class_name = "BPD"
    elif predicted_classes[0] == 2:
        predicted_class_name = "Bipolar"
    elif predicted_classes[0] == 3:
        predicted_class_name = "Depression"
    elif predicted_classes[0] == 4:
        predicted_class_name = "Schizophrenia"

    # Get the confidence score for the predicted class
    predicted_confidence = prediction[0][predicted_classes[0]]  # Confidence of the predicted class

    # Round the confidence value to two decimal places and convert to percentage
    rounded_confidence = round(predicted_confidence * 100, 2)  # Convert to percentage

    # Format the confidence value to include the '%' symbol
    confidence_percentage = f"{rounded_confidence}%"
    print(predicted_class_name, confidence_percentage)

    return predicted_class_name, confidence_percentage


def clear_fields():
    # Clear all input fields
    for entry in entries:
        entry.delete(0, tk.END)

    # Reset predicted states
    global predicted_class_name, confidence_percentage
    predicted_class_name = ''
    confidence_percentage = ''

def submit():
    # Collect the input from the GUI and validate
    inputs = [entry.get().strip() for entry in entries]

    # Check if any field is empty
    if any(not input_text for input_text in inputs):
        # Show an error message if any field is empty
        messagebox.showerror("Missing Answer", "Please fill in all fields before submitting.")
        return
        # Display "Processing..." while the model runs
    result_label.config(text="Processing...", fg="blue")
    root.update_idletasks()  # Force immediate update of the GUI

    # Call the prediction function with the input list
    predicted_class_name, confidence_percentage = make_prediction(inputs)

    # Reset the "Processing..." label after prediction
    result_label.config(text="")  # Clear the label text

    # Create a new popup window using Toplevel
    popup = tk.Toplevel(root)
    popup.title("Prediction Result")

    # Set the size of the popup
    popup.geometry("400x200")  # Customize the size as needed

    # Display the message in the popup window
    if float(confidence_percentage[:-1]) < 50:  # Confidence is below 50%
        message = (f"The model is not highly confident.\nPredicted class: {predicted_class_name}"
                   f"\nConfidence: {confidence_percentage}\nPlease review the result carefully.")
    else:
        message = f"Predicted class: {predicted_class_name}\nConfidence: {confidence_percentage}"

    label = tk.Label(popup, text=message, font=("Helvetica", 14), padx=20, pady=20)
    label.pack()

    # Add a button to close the popup
    close_button = tk.Button(popup, text="Close", command=popup.destroy, font=("Helvetica", 12))
    close_button.pack(pady=10)


def exit_app():
    root.quit()


# Create main window
root = tk.Tk()
root.title("Questionnaire")

# Get the screen width and height
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Get the window width and height
window_width = 900  # Same as the size defined in geometry
window_height = 700  # Same as the size defined in geometry

# Calculate the x and y position to center the window
x_position = int((screen_width - window_width) / 2)
y_position = int((screen_height - window_height) / 2)

# Set the position of the window
root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")

questions = [
    "Question 1",
    "Question 2",
    "Question 3",
    "Question 4",
    "Question 5",
]

entries = []  # To hold entry widgets

question_font = ("Helvetica", 16)  # Font for questions
entry_font = ("Helvetica", 14)  # Font for input fields

root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)

# Create GUI elements
for i, question in enumerate(questions):
    tk.Label(root, text=question, font=question_font).grid(
        row=2 * i, column=0, columnspan=2, pady=(15, 5), sticky="n"
    )
    entry = tk.Entry(root, font=entry_font, width=60)
    entry.grid(row=2 * i + 1, column=0, columnspan=2, pady=(5, 20), sticky="n")
    entries.append(entry)

# Submit and exit buttons
button_frame = tk.Frame(root)
button_frame.grid(row=2 * len(questions), column=0, columnspan=2, pady=20)

submit_button = tk.Button(
    button_frame, text="Submit", command=submit, font=("Helvetica", 16), width=15
)
submit_button.pack(side=tk.LEFT, padx=10)

exit_button = tk.Button(
    button_frame, text="Exit", command=exit_app, font=("Helvetica", 16), width=15
)
exit_button.pack(side=tk.LEFT, padx=10)

clear_button = tk.Button(
    button_frame, text="Clear", command=clear_fields, font=("Helvetica", 16), width=15
)
clear_button.pack(side=tk.LEFT, padx=10)

# Label for displaying results
result_label = tk.Label(root, text="", font=("Helvetica", 14), fg="blue")
result_label.grid(row=2 * len(questions) + 1, column=0, columnspan=2, pady=10, sticky="n")

# Run the application
root.mainloop()

