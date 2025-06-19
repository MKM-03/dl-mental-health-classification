import tensorflow as tf
import numpy as np

# Load embeddings and labels
embeddings = np.load('embeddings_dir/padded_embeddings_large_100.npy')
labels = np.load('labels/encoded_labels_test.npy')


# Define a function to convert data to tf.train.Example
def serialize_example(embedding, label):
    feature = {
        'embedding': tf.train.Feature(float_list=tf.train.FloatList(value=embedding.flatten())),
        'label': tf.train.Feature(float_list=tf.train.FloatList(value=label))
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


# Write embeddings and labels to TFRecord
with tf.io.TFRecordWriter('embeddings_dir/embeddings_with_labels_large_100.tfrecord') as writer:
    for embedding, label in zip(embeddings, labels):
        example = serialize_example(embedding, label)
        writer.write(example)
