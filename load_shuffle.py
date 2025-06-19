import tensorflow as tf


feature_description = {
    'embedding': tf.io.FixedLenFeature([10000], tf.float32),  # Flattened shape of (100, 100)
    'label': tf.io.FixedLenFeature([5], tf.float32)           # One-hot encoded label shape (5,)
}


# Parsing function to reshape embedding
def parse_tfrecord_fn(example_proto):
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)
    parsed_example['embedding'] = tf.reshape(parsed_example['embedding'], (100, 100))  # Reshape back to (100, 200)
    return parsed_example['embedding'], parsed_example['label']


# Load the TFRecord dataset
dataset = tf.data.TFRecordDataset('embeddings_dir/embeddings_with_labels_large_100.tfrecord')
dataset = dataset.map(parse_tfrecord_fn)

dataset_size = sum(1 for _ in dataset)
print(dataset_size)

# Shuffle once for consistent splits
shuffled_dataset = dataset.shuffle(buffer_size=dataset_size, reshuffle_each_iteration=False)

# Save the shuffled dataset
tf.data.experimental.save(shuffled_dataset, 'ready_ds/shuffled_dataset_large_100')
