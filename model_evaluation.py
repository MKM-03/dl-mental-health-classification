import tensorflow as tf
import tensorflow_addons as tfa

test_dataset = tf.data.experimental.load('ready_ds/dataset_eval_large_100')
test_dataset = test_dataset.batch(64)

cnn_lstm = tf.keras.models.load_model('models/checkpoint_large')
# cnn = tf.keras.models.load_model('models/best_enhanced_first')

cnn_lstm.summary()

test_loss, test_accuracy, test_precision, test_recall, test_f1 = cnn_lstm.evaluate(test_dataset)

# Evaluation of CNN-LSTM model
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')
print(f'Test Precision: {test_precision}')
print(f'Test Recall: {test_recall}')
print(f'Test F1: {test_f1}')

# test_loss, test_accuracy, test_precision, test_recall, test_f1 = cnn.evaluate(test_dataset)
#
# # Evaluation of CNN model
# print(f'Test Loss: {test_loss}')
# print(f'Test Accuracy: {test_accuracy}')
# print(f'Test Precision: {test_precision}')
# print(f'Test Recall: {test_recall}')
# print(f'Test F1: {test_f1}')
