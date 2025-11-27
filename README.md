# Multi-Image Classification using Deep Learning

This repository contains a complete, end-to-end pipeline for performing multi-class image classification using a Convolutional Neural Network (CNN). The project covers everything from data loading and preprocessing to model training, evaluation, and making predictions.

## Project Overview

This project implements a robust deep learning model designed to accurately classify images into several distinct categories. It is structured as a self-contained Jupyter Notebook (multi_image_classification.ipynb) that guides the user through the entire process.

## Key Features

- Convolutional Neural Network (CNN) Architecture: Implementation of a custom or pre-trained CNN model optimized for image feature extraction and classification.

![licensed-image](https://github.com/user-attachments/assets/c3c9f803-c7ec-48ff-ab2a-3dbed2bc6c32)


- Data Augmentation: Techniques applied to increase the diversity of the training set and prevent overfitting.

- Transfer Learning (Optional): Utilizes pre-trained weights from state-of-the-art models (e.g., VGG, ResNet) for accelerated training and improved performance on smaller datasets. (Note: Adjust this if you used a custom-built CNN only).

- Model Training & Evaluation: Detailed training history tracking and metric analysis (accuracy, loss).

## Technologies & Libraries

This project is built using the foundational tools of the Python machine learning ecosystem:

- Python

- TensorFlow / Keras: Primary framework for building and training the deep learning model.

- NumPy: For numerical operations and data handling.

- Pandas: For data manipulation and analysis.

- Matplotlib / Seaborn: For visualizing images, training history, and final results.

## Dataset

*(Note to User: Replace the placeholder text below with the actual name and source of the dataset you used, e.g., "CIFAR-10," "Kaggle Birds Dataset," or a description of the data's content.)*

The model was trained on the Intel Image Classification dataset, consisting of 6 distinct classes. The data is preprocessed to ensure uniformity in size and normalized pixel values before being fed into the network.

## Installation and Setup

To run this notebook locally, ensure you have Python installed, and then install the required libraries:

  Clone the repository:
    
    Bash

    git clone https://github.com/muneebsaddal/multi-image-classification.git
    cd multi-image-classification

Install dependencies:
    
    Bash

    pip install tensorflow keras numpy pandas matplotlib

(If you used a GPU, you may need to install the appropriate GPU versions of TensorFlow.)

Run the notebook:

    Bash

    jupyter notebook multi_image_classification.ipynb

## Results

| Metric | Training Result | Test Result |
|--------|--------|--------|
| Accuracy | 90.49% | 85.60% |
| Loss | 0.2368 | 0.4531 |
