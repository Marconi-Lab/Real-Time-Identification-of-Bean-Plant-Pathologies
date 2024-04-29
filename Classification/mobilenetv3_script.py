import argparse
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small  
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_model(num_classes):
    """Creates and returns the MobileNetV3 model."""
    base_model = MobileNetV3Small(include_top=False, input_shape=(224, 224, 3), weights='imagenet')
    base_model.trainable = True  
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=True)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    return model

def train_model(train_dir, validation_dir, batch_size, epochs):
    """Train the model and save it."""
    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = val_datagen.flow_from_directory(
        validation_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical')

    model = create_model(train_generator.num_classes)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_generator, epochs=epochs, validation_data=validation_generator)
    model.save('mobilenetv3_model.h5')

def evaluate_model(model_path, test_dir, batch_size):
    """Load the model and evaluate it on the test set."""
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical')

    model = tf.keras.models.load_model(model_path)
    results = model.evaluate(test_generator)
    print("Test results - Loss: {:.2f}, Accuracy: {:.2%}".format(results[0], results[1]))

def main():
    parser = argparse.ArgumentParser(description="Train or evaluate MobileNetV3 on a dataset.")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'evaluate'], help='Mode to run the script in.')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing training/validation/test data.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training or evaluation.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training.')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_model(args.data_dir + '/train', args.data_dir + '/val', args.batch_size, args.epochs)
    elif args.mode == 'evaluate':
        evaluate_model('mobilenetv3_model.h5', args.data_dir + '/test', args.batch_size)

if __name__ == "__main__":
    main()
