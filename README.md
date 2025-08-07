# MNIST Handwritten Digit Classification using CNN
In this project, I've built a CNN (Convolutional Neural Network) using TensorFlow and Keras to classify handwritten digits from the MNIST dataset.

## 📚 What is MNIST?
* MNIST is a standard benchmark dataset in machine learning.
* It contains 70,000 grayscale images of handwritten digits (0–9).
  - 60,000 images are used for training.
  - 10,000 images are used for testing.
* Each image is 28x28 pixels in size and contains a single digit in the center.

## Goal of my project 
* The goal is to train a CNN that takes a 28x28 grayscale image and predicts which digit (0–9) it is.
* The model should be able to generalize well on unseen test data.

# Steps that I followed:
##  Step 1: Data Preprocessing

* The MNIST dataset is directly imported from `keras.datasets`, so no need to manually download or clean it.
* Images are normalized by dividing pixel values by 255 to bring them between 0 and 1.
* Each image is reshaped from 28x28 to 28x28x1 to match the expected input shape of CNN (last `1` indicates grayscale).
* The labels (digits 0–9) are one-hot encoded, converting them into a format suitable for categorical classification.


## Step 2: CNN Model Architecture

The CNN used in this project follows a standard layered structure:

* First Convolution Layer:
  - Detects 32 features from the image using 3x3 filters.
  - Followed by a ReLU activation for non-linearity.

* First MaxPooling Layer:
  - Reduces the size of the feature maps to retain only important features and reduce computation.

* Second Convolution Layer:
  - Detects 64 deeper features from the previous layer's output.

* Second MaxPooling Layer:
  - Again reduces dimensionality and keeps only the most useful features.

* Flatten Layer:
  - Converts the 2D output into a 1D vector to feed into fully connected (dense) layers.

* Dense Layer:
  - A fully connected layer with 128 neurons to learn patterns in the data.

* Output Layer:
  - A softmax layer with 10 neurons (one for each digit from 0 to 9).
  - Outputs a probability distribution over the 10 classes.

---

## Step 3: Compiling the Model

* The model is compiled using the Adam optimizer, which adapts the learning rate automatically.
* The loss function is categorical cross-entropy since it's a multi-class classification task.
* Accuracy is used as the metric to evaluate model performance during training.

---

## 🏋️ Step 4: Training the Model

* The model is trained on the training set (`x_train`, `y_train`).
* Validation data is provided to monitor performance on unseen data.
* Epochs define how many times the entire dataset is passed through the network.

---

## 🧪 Step 5: Evaluating the Model

* The trained model is evaluated on the test data (`x_test`, `y_test`).
* The final test accuracy tells how well the model performs on unseen handwritten digits.

## Conclusion

* This project helped me in understanding and implementing image classification using CNN.
* I used TensorFlow and Keras for model building, training, and evaluation.
* The project demonstrates how preprocessing, model design, training, and evaluation work together in a real-world classification task.
