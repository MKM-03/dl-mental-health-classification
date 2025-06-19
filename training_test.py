import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv1D, LSTM, Dense, Dropout, Bidirectional,
                                     BatchNormalization, SpatialDropout1D, InputLayer)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


train_dataset = tf.data.experimental.load('ready_ds/dataset_train_large_100')
val_dataset = tf.data.experimental.load('ready_ds/dataset_val_large_100')
BATCH_SIZE = 64

print(len(train_dataset))

train_dataset = train_dataset.shuffle(40000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE)

model = Sequential([
    InputLayer(input_shape=(100, 100)),

    Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),
    SpatialDropout1D(0.1),

    Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),
    SpatialDropout1D(0.1),

    Conv1D(filters=256, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),
    SpatialDropout1D(0.1),

    Conv1D(filters=256, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),
    SpatialDropout1D(0.1),

    Bidirectional(LSTM(128, return_sequences=True)),
    Bidirectional(LSTM(64, return_sequences=False)),
    Dropout(0.5),

    Dense(5, activation='softmax')
])

model.summary()

f1_metric = tfa.metrics.F1Score(num_classes=5, average='macro')

optimizer = Adam(learning_rate=0.001, clipnorm=1.0)

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model_checkpoint = ModelCheckpoint('models/best_testing_prototype', monitor='val_loss', save_best_only=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.2, min_lr=1e-6)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy',
                                                                             tf.keras.metrics.Precision(),
                                                                             tf.keras.metrics.Recall(),
                                                                             f1_metric])

epochs = 100

history = model.fit(train_dataset,
                    validation_data=val_dataset,
                    callbacks=[early_stopping, model_checkpoint, reduce_lr],
                    epochs=epochs)

tf.keras.models.save_model(model, 'models/testing_prototype')

# Plot Training and Validation Metrics
plt.figure(figsize=(18, 6))

# Loss
plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Accuracy
plt.subplot(1, 3, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# F1 Score
plt.subplot(1, 3, 3)
plt.plot(history.history['f1_score'], label='Training F1 Score')
plt.plot(history.history['val_f1_score'], label='Validation F1 Score')
plt.title('Training and Validation F1 Score')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.legend()

plt.tight_layout()
plt.show()
