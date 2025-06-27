# 🧠 DL Mental Health Classification

## A beginner-friendly deep learning project focused on **mental health classification**.
Whether you're just getting started with AI or looking to understand how deep learning can be applied to real-world problems, this project will help you understand how to prepare data, preprocess data, create and train a DL model.<br>
Below is how the project was approached:

### - **Data preparation**
  > This includes the following files in that order:
>  - Dataset https://www.kaggle.com/datasets/kamaruladha/mental-disorders-identification-reddit-nlp
>   - data_preprocessing.py
>   - one_hot_labels.py
>   - train_fasttext.py
>   - generate_embeddings.py
>   - convert_TFRecord.py
>   - load_shuffle.py
>   - create_dataset.py

### - **Model creation training and testing**
  > This includes creation_training.py

### - **Model evaluation**
  > This includes model_evaluation.py

### - **Demostration**
  > The purpose of this is to show a simple way to use the model created for classifiying the mental disorders:
>  - main_test.py
>  - application.py

# IMPORTANT NOTES!
  This project uses tensorflow keras framework. Please use requirements.txt to install the necessary libararies for the project to work as intended, the versions must be the same as the ones in the text file.<br>
  Also do note that to use the full dataset, same as i did, you will need an Nvidia gpu of 8gb VRAM as minimum, if you dont have a gpu with that much you wont able to fit the entire dataset into your gpu to proccess
  However you can always undersample the dataset.<br>
  my device's spec:
  - GPU: RTX 3050 8GB VRAM
  - CPU: R5 5600 6 cores/12 threads
  - 16gb DDR4 RAM 3600mhz
<br> One last important thing to note, you will need to go to https://www.tensorflow.org/install/source_windows#gpu and install the version of CUDA and CUDnn that is compatible with the tensorflow version in this project (i used TF 2.9 or 2.10)
