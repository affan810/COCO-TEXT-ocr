import os
import cv2
import numpy as np

import tensorflow as tf
from keras import layers, models

def build_cnn_model(input_shape, num_classes):
    model = models.Sequential()
    
    # Convolutional layers
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    
    # Flatten layer
    model.add(layers.Flatten())
    
    # Dense layers
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model

def train_model(model, train_images, train_labels, epochs, batch_size):
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size)


import os
import cv2
import numpy as np

def preprocess_image(image_path, target_size):
    try: 
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale
        img = cv2.resize(img, target_size)  # Resize image to target size
        img = img.astype('float32') / 255.0  # Normalize pixel values to range [0, 1]
        img = np.expand_dims(img, axis=-1)  # Add channel dimension
        return img
    except: return -1

def load_images_from_folder(folder_path, target_size):
    images = []
    labels = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".jpeg"):
            image_path = os.path.join(folder_path, filename)
            label = filename.split('.')[0]  # Assuming filename contains label
            print(filename)
            ans = preprocess_image(image_path, target_size)
            if not isinstance(ans, int):
                images.append(ans)
                labels.append(label)
    return np.array(images), np.array(labels)

def convert_labels_to_one_hot(labels, max_string_length, char_to_index):
    num_classes = len(char_to_index)
    one_hot_labels = np.zeros((len(labels), max_string_length, num_classes), dtype=np.uint8)
    for i, label in enumerate(labels):
        for j, char in enumerate(label):
            one_hot_labels[i, j, char_to_index[char]] = 1
    return one_hot_labels

def main():
    folder_path = "images\\"
    target_size = (32, 32)  # Target size for the images
    
    train_images, train_labels = load_images_from_folder(folder_path, target_size)
    print(train_labels)
    # Assuming you have a dictionary mapping characters to indices
    char_to_index = {
    'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9,
    'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19,
    'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25,
    'a': 26, 'b': 27, 'c': 28, 'd': 29, 'e': 30, 'f': 31, 'g': 32, 'h': 33, 'i': 34, 'j': 35,
    'k': 36, 'l': 37, 'm': 38, 'n': 39, 'o': 40, 'p': 41, 'q': 42, 'r': 43, 's': 44, 't': 45,
    'u': 46, 'v': 47, 'w': 48, 'x': 49, 'y': 50, 'z': 51,
    '0': 52, '1': 53, '2': 54, '3': 55, '4': 56, '5': 57, '6': 58, '7': 59, '8': 60, '9': 61
    }

    # Convert labels to one-hot encoding
    train_labels_one_hot = []
    for j in range(len(train_labels)):
        toh = []
        for i in range(62):
            if list(char_to_index.keys())[i] in train_labels[j]:
                toh.append(1)
            else:
                toh.append(0)
        train_labels_one_hot.append(toh)
    # print(train_labels_one_hot)
    train_labels_one_hot = np.array(train_labels_one_hot)
    input_shape = train_images[0].shape
    num_classes = 62
    test_images, test_labels = train_images, train_labels_one_hot
    # Build and train the model
    model = build_cnn_model(input_shape, num_classes)
    train_model(model, train_images, train_labels_one_hot, epochs=10000, batch_size=256)
    
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('Test accuracy:', test_acc)
    # Now train_labels_one_hot can be used as the labels for training your model
    
if __name__ == "__main__":
    main()


#Epoch 1577/10000
#70/70 [==============================] - 8s 112ms/step - loss: 5331260303713763328.0000 - accuracy: 0.0099
#full dataset batch_size=256