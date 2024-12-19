# AI-Medical-Imaging
construction of a convolutional  network to classify SPECT images into its four stages of Parkinson's disease. I already processed the images that come from a tomography process and had to be 3D rendered and colored for it to be sliced into coronal photograms. Each stage has around 50 patients and each patient has around 30 images, except for the fourth stage which has no more than 15 patients data available at the foundation I am getting the data from. Referring to this issue, I asked my advisor about this lack of data and told me that it is no big issue since I can compensate for it with data from the other stages.
----------------
To build a convolutional neural network (CNN) for classifying SPECT images of Parkinson's disease into four stages, you'll need to follow several steps, from preparing the data to constructing and training the model. Since you're dealing with 3D medical images, this problem will be a bit more complex than standard 2D image classification tasks, but a 3D convolutional neural network (3D CNN) can be very effective for this type of data.
Steps to Approach the Problem:

    Data Preparation:
        You need to organize your data and preprocess it (normalization, resizing, augmentation, etc.).
        As you mentioned, there is an imbalance in the dataset, especially for Stage 4. Data augmentation techniques can help compensate for the small number of samples in this stage.

    Building the Model:
        For a 3D CNN, you will use Conv3D layers instead of Conv2D to process the three-dimensional data.
        You'll also need to use pooling layers (e.g., MaxPooling3D), batch normalization layers, and dropout layers to avoid overfitting.

    Training:
        Since you have class imbalance, you might want to use techniques like class weights, oversampling, or data augmentation to address this issue.

    Evaluation:
        You can use metrics like accuracy, F1-score, or AUC (area under the curve) to evaluate the performance of your model.

Here's how to implement this process:
Python Code for 3D CNN to Classify SPECT Images into Parkinson's Disease Stages

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Step 1: Data Preprocessing
# Assuming `data` is a NumPy array containing the images (3D array for each image).
# `labels` should be a 1D array of shape (num_samples,) containing the labels (stages of Parkinson's disease)

# Placeholder for the data loading process
# data = np.load('path_to_processed_images.npy')  # Shape should be (num_samples, height, width, depth, channels)
# labels = np.load('path_to_labels.npy')  # Labels should be (num_samples,)

# Normalize the images to [0, 1]
data = data.astype('float32') / 255.0

# Step 2: Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42, stratify=labels)

# Step 3: Define a simple 3D CNN model for image classification

def create_3d_cnn(input_shape, num_classes):
    model = models.Sequential()

    # First convolution block
    model.add(layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))

    # Second convolution block
    model.add(layers.Conv3D(64, kernel_size=(3, 3, 3), activation='relu'))
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))

    # Third convolution block
    model.add(layers.Conv3D(128, kernel_size=(3, 3, 3), activation='relu'))
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))

    # Flatten the output
    model.add(layers.Flatten())

    # Fully connected layers
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

# Step 4: Model compilation
input_shape = X_train.shape[1:]  # (height, width, depth, channels)
num_classes = len(np.unique(y_train))  # Number of stages of Parkinson's disease

model = create_3d_cnn(input_shape, num_classes)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Step 5: Data Augmentation (optional)
# Use data augmentation to increase the size of the training set, especially for Stage 4 (underrepresented class).
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

datagen.fit(X_train)

# Step 6: Training the model with class weights (to handle class imbalance)
class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)

# Step 7: Train the model
history = model.fit(datagen.flow(X_train, y_train, batch_size=16),
                    validation_data=(X_val, y_val),
                    epochs=20,
                    class_weight={i: class_weights[i] for i in range(num_classes)})

# Step 8: Evaluate the model on the validation set
y_pred = np.argmax(model.predict(X_val), axis=1)

# Step 9: Print the classification report and confusion matrix
print(classification_report(y_val, y_pred))
print(confusion_matrix(y_val, y_pred))

# Step 10: Plot the training and validation loss/accuracy curves
plt.figure(figsize=(12, 4))

# Loss Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Accuracy Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# Step 11: Saving the model
model.save('parkinsons_spect_model.h5')

Key Points in the Code:

    Data Preprocessing: The images are normalized to a range between 0 and 1. This step is important for deep learning models to perform well.

    3D CNN Architecture: The model uses 3D convolutional layers (Conv3D and MaxPooling3D) to process the 3D medical images. The architecture includes:
        3 convolutional blocks, each followed by max-pooling layers.
        A fully connected layer after flattening the output of the convolutional layers.
        A dropout layer to prevent overfitting.
        The output layer uses softmax activation to classify the images into one of the 4 stages of Parkinson’s disease.

    Data Augmentation: We use ImageDataGenerator to apply data augmentation, which helps generate more training data from the existing dataset, especially for the minority class (Stage 4).

    Class Weight Adjustment: class_weight is computed to address the class imbalance. It adjusts the loss function during training to give more importance to underrepresented classes.

    Model Evaluation: After training, we evaluate the model using accuracy and loss curves and generate a classification report and confusion matrix for a detailed performance analysis.

    Saving the Model: Finally, we save the trained model to disk for later use.

Handling Imbalanced Data:

Since Stage 4 has significantly fewer samples, it's important to address this imbalance. The data augmentation technique helps increase the variety of images for training, and class weights help ensure that the model doesn't bias towards the majority classes.
Further Improvements:

    Advanced Techniques: You can try more advanced models like 3D ResNet or 3D DenseNet for better performance.
    Cross-validation: You can use cross-validation techniques to better evaluate your model.
    Transfer Learning: Using pre-trained 3D models (e.g., 3D CNNs pre-trained on other medical datasets) could also improve accuracy, especially when data is limited.

This implementation provides a strong starting point for building a 3D CNN for classifying Parkinson's disease stages from SPECT images.
