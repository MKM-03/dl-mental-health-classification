# 🧠 DL Mental Health Classification

## Abstract
Mental health has always been an important matter for the well-being of people of different ages, especially adolescents as the next working force for different societies, it affects their behaviour, connections with other people, personal life matters, and their work. Furthermore, it is crucial to maintain mental wellness by identifying mental disorders early and taking measures against them. AI and technological advancements play an important role in assisting individuals with mental disorders and aiding mental health professionals with the diagnosis process. Several individuals do not wish to interact with a mental health professional for either financial or personal reasons, that is when the use of deep learning to develop a model that can help some individuals get insights on chances for having one or several mental disorders, which could be particularly useful for adolescents and mental health professionals to help make the diagnosis process easier for both parties.

## A beginner-friendly deep learning project focused on **mental health classification**.
Whether you're just getting started with AI or looking to understand how deep learning can be applied to real-world problems, this project will help you understand how to prepare data, preprocess data, create and train a DL model.<br>
Below is how the project was approached:

### - **Exploratory Data Analysis**
  > This would include a jupyter notebook file that i made quickly to explain a couple of things for those who are interested:

### - **Data Preprocessing**
  > This includes the following files in that order:
>  - Dataset https://www.kaggle.com/datasets/kamaruladha/mental-disorders-identification-reddit-nlp
>   - data_preprocessing.py
>   - one_hot_labels.py
>   - train_fasttext.py
>   - generate_embeddings.py
>   - convert_TFRecord.py
>   - load_shuffle.py
>   - create_dataset.py

### - **Model Creation Training and Testing**
  > This includes creation_training.py

### - **Model evaluation**
  > This includes model_evaluation.py

### - **Demostration**
  > The purpose of this is to show a simple way to use the model created for classifiying the mental disorders:
>  - main_test.py
>  - application.py<br>
# IMPORTANT NOTES!
  This project uses tensorflow keras framework. Please use requirements.txt to install the necessary libararies for the project to work as intended, the versions must be the same as the ones in the text file.<br>
  Also do note that to use the full dataset, same as i did, you will need an Nvidia gpu of 8gb VRAM as minimum, if you dont have a gpu with that much you wont able to fit the entire dataset into your gpu to proccess
  However you can always undersample the dataset.<br>
  my device's spec:
  - GPU: RTX 3050 8GB VRAM
  - CPU: R5 5600 6 cores/12 threads
  - 16gb DDR4 RAM 3600mhz<br>
One last important thing to note, you will need to go to https://www.tensorflow.org/install/source_windows#gpu and install the version of CUDA and CUDnn that is compatible with the tensorflow version in this project (i used TF 2.9 or 2.10)
