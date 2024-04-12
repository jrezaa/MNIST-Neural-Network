# MNIST Image Classification using Convolutional Neural Network

This project implements a simple convolutional neural network (CNN) to classify grayscale images of handwritten digits (0 - 9) from the MNIST dataset. The dataset consists of 60,000 training images and 10,000 testing images, each with a size of 28x28 pixels and a single channel.

## Setup
The project contains the following files:
- `src/mnist/nn.py`: Contains code for the neural network implementation.
- `src/mnist/images_train.csv.gz`: Training images data.
- `src/mnist/labels_train.csv.gz`: Training labels data.
- `src/mnist/images_test.csv.gz`: Testing images data.
- `src/mnist/labels_test.csv.gz`: Testing labels data.

## Requirements
- Python 3.x
- NumPy
- Matplotlib

## Usage
1. **Unregularized Model**:
   - Implemented forward-propagation and back-propagation for the loss function specified.
   - Initialized weights from a standard normal distribution and bias/intercept terms to 0.
   - Trained the model using mini-batch gradient descent with a batch size of 1,000 for 30 epochs.
   - Ploted loss and accuracy over epochs for both training and dev sets.
   - Saved learnt parameters at the end of 30 epochs.

2. **Regularized Model**:
   - Added a regularization term to the cross entropy loss.
   - Implemented the regularized version and repeat training and plotting as in the unregularized model.
   - Saved learnt parameters separately.

3. **Final Test**:
   - Evaluated the performance of both models on the test set using the saved parameters.
   - Reported test accuracy for both regularized and non-regularized models.

## Observations
- Regularization typically improves generalization by reducing overfitting, hence, it's expected to see better performance on the test set with the regularized model.
- The reported accuracies align with this expectation, with the regularized model achieving higher accuracy due to reduced overfitting.

## Note
Parameters learnt during training are saved for future use, eliminating the need for re-training.
