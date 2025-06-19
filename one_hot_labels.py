import pandas as pd
import numpy as np
from sklearn import preprocessing
import tensorflow as tf


# Reads the dataset file into a pandas dataframe with a specified encoding
def read_dataset(path, encoding):
    return pd.read_csv(path, encoding=encoding)


def encode_labels(df, classes):
    le = preprocessing.LabelEncoder()
    df['label'] = le.fit_transform(df['label'])

    labels = df.label.to_numpy(dtype='int32')

    encoded_l = tf.keras.utils.to_categorical(labels, num_classes=classes)
    encoded_l = encoded_l.astype('int32')

    # Creating a mapping of label integer values to their corresponding class names
    class_mapping = {index: label for index, label in enumerate(le.classes_)}

    return encoded_l, class_mapping


# Read CSV file into pandas DataFrame
data_file = 'data/cleaned_reddit.csv'
main_data = read_dataset(data_file, 'utf-8')
print(main_data.head(5))
print(main_data['label'].value_counts())

# Remove mentalillness class
main_data = main_data[main_data.label != 'mentalillness']

main_data = main_data.drop_duplicates().dropna()
print(main_data.label.value_counts())

# Get one-hot encoded labels and class mapping
OH_LABELS, class_mapping = encode_labels(main_data, 5)

# Save the one-hot encoded labels
np.save('labels/encoded_labels_test.npy', OH_LABELS)

# Print the class mappings
print("Class mapping:")
for index, label in class_mapping.items():
    print(f"Index: {index}, label: {label}")
