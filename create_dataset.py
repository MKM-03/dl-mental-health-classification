import tensorflow as tf

# Load the shuffled dataset for splitting
dataset = tf.data.experimental.load('ready_ds/shuffled_dataset_large_100')

# Get dataset size
dataset_size = len(dataset)
print(dataset_size)

# Calculate split sizes
train_size = int(0.7 * dataset_size)
val_size = int(0.2 * dataset_size)
eval_size = dataset_size - train_size - val_size

# Create separate splits with independent takes
train_dataset = dataset.take(train_size)
print(len(train_dataset))

val_dataset = dataset.skip(train_size).take(val_size)
print(len(val_dataset))

eval_dataset = dataset.skip(train_size + val_size).take(eval_size)
print(len(eval_dataset))

tf.data.experimental.save(train_dataset, 'ready_ds/dataset_train_large_100')
tf.data.experimental.save(val_dataset, 'ready_ds/dataset_val_large_100')
tf.data.experimental.save(eval_dataset, 'ready_ds/dataset_eval_large_100')
