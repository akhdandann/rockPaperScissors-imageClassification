# Rock-Paper-Scissors Image Classifier using CNN and Data Augmentation

This repository contains a TensorFlow + Keras implementation of a **Convolutional Neural Network (CNN)** model to classify images of hand gestures representing **Rock**, **Paper**, and **Scissors**.

## 📌 Description

This project is a practical image classification task where the model learns to distinguish between three categories of hand gestures: rock, paper, and scissors. The dataset is obtained from **Dicoding Academy's Rock-Paper-Scissors image set**, and the model uses **image augmentation** to improve generalization.

The model is trained on preprocessed images using `ImageDataGenerator` with transformations such as rotation, flipping, and zooming to simulate real-world variance.

The classification is performed using a **multi-class CNN** with 3 output nodes and `softmax` activation.

## 📁 Dataset

The dataset used is from Dicoding and publicly available:

- [Rock Paper Scissors Dataset](https://github.com/dicodingacademy/assets/releases/download/release/rockpaperscissors.zip)

It consists of labeled folders:
- `rock/`
- `paper/`
- `scissors/`

The code automatically downloads and extracts the dataset for training and validation.

## 🛠 Features

- Automatic dataset download and unzip
- Data augmentation using `ImageDataGenerator`
- Training-validation split with shuffling
- Convolutional Neural Network with 3 Conv layers
- Accuracy and loss graph visualization
- Image classification on user-uploaded image

## ⚙️ How It Works

1. **Download & Extract Data**
   - Uses `wget` and `zipfile` to fetch and unzip the dataset

2. **Data Exploration**
   - Verifies and prints folder structure

3. **Data Augmentation**
   - Applies rotation, zoom, flip, shear, etc. using `ImageDataGenerator`

4. **Model Architecture**
   - 3 convolutional layers + max pooling
   - Fully connected layer (512 neurons)
   - Output layer with 3 neurons + softmax

5. **Training**
   - Uses RMSprop optimizer and categorical crossentropy
   - Trains for 50 epochs with validation split

6. **Visualization**
   - Plots training & validation accuracy and loss

7. **Image Prediction**
   - Uploads image using `files.upload()` and classifies using the trained model

## 🚀 How to Run

1. Open the notebook in **Google Colab** or any Jupyter environment  
2. Install dependencies:
   ```python
   !pip install wget keras_preprocessing
   ```
3. Run the cells step by step:
   - Download and extract dataset  
   - Train the model  
   - Upload an image to test prediction  

## 📊 Output

- CNN model summary  
- Accuracy and loss charts  
- Prediction result of uploaded image (rock, paper, or scissors)  

## 📎 Dependencies

- TensorFlow  
- Keras Preprocessing  
- Numpy  
- Pandas  
- Matplotlib  
- wget  
- Google Colab (for `files.upload()`)

## 🧾 Note

This project was originally created as part of a **Dicoding certification test** around 5 years ago.  
While it may be slightly dated, the structure and techniques are still **very useful for those preparing for upcoming certification exams or learning basic CNNs with image data**.

Feel free to explore, learn, or build upon it!
