import argparse
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import ImageFile
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pandas as pd

def create_model(input_shape, n_classes):
    """Creates and returns the model."""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), strides=(2, 2), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.DepthwiseConv2D((3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (1, 1), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'),
        layers.DepthwiseConv2D((3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (1, 1), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Flatten(),
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(n_classes, activation='softmax')
    ])
    return model

def train_model(train_dir, validation_dir, image_size, batch_size, epochs):
    """Trains the model and returns the trained model and history."""
    image_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    train_ds = image_datagen.flow_from_directory(train_dir, class_mode='categorical', batch_size=batch_size, target_size=(image_size, image_size))
    validation_ds = image_datagen.flow_from_directory(validation_dir, class_mode='categorical', batch_size=batch_size, target_size=(image_size, image_size))

    model = create_model((image_size, image_size, 3), 4)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10, restore_best_weights=True, verbose=1)
    history = model.fit(train_ds, validation_data=validation_ds, epochs=epochs, callbacks=[early_stopping], workers=10, use_multiprocessing=True)
    return model, history

def evaluate_model(model, test_dir, image_size, batch_size):
    """Evaluates the model and prints the results."""
    image_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    test_ds = image_datagen.flow_from_directory(test_dir, class_mode='categorical', batch_size=batch_size, target_size=(image_size, image_size))
    results = model.evaluate(test_ds)
    print("Test Loss, Test Accuracy:", results)

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate Bean Disease Classifier.")
    parser.add_argument('--mode', type=str, required=True, help='Mode: "train" or "evaluate"')
    parser.add_argument('--train_dir', type=str, default='*****/training', help='Directory with training data')
    parser.add_argument('--test_dir', type=str, default='*****/test', help='Directory with test data')
    parser.add_argument('--validation_dir', type=str, default='*****/validation', help='Directory with validation data')
    parser.add_argument('--image_size', type=int, default=224, help='Target size of the images')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and evaluation')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        model, history = train_model(args.train_dir, args.validation_dir, args.image_size, args.batch_size, args.epochs)
        model.save('bean_disease_model.h5')  
        print("Model trained and saved as bean_disease_model.h5")
    elif args.mode == 'evaluate':
        model = tf.keras.models.load_model('bean_disease_model.h5')  
        evaluate_model(model, args.test_dir, args.image_size, args.batch_size)
    else:
        raise ValueError("Unsupported mode. Use 'train' or 'evaluate'.")

if __name__ == "__main__":
    main()


