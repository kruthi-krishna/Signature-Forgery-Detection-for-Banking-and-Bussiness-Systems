import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def build_model(input_shape=(128, 128, 1), num_classes=2):
    """Builds and compiles a CNN model for image classification."""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(data_dir, model_path, img_size=(128, 128), batch_size=32, epochs=10):
    """
    Trains the CNN model on the provided dataset and saves the model to a file.

    Args:
    - data_dir (str): Path to the dataset directory.
    - model_path (str): Path to save the trained model.
    - img_size (tuple): Image size for resizing (default: (128, 128)).
    - batch_size (int): Number of images per batch (default: 32).
    - epochs (int): Number of training epochs (default: 10).
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset directory '{data_dir}' does not exist.")

    # Data augmentation and rescaling
    datagen = ImageDataGenerator(
        rescale=1.0/255,
        validation_split=0.2  # Reserve 20% of data for validation
    )

    # Training and validation generators
    train_gen = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='binary',
        subset='training',
        shuffle=True
    )
    val_gen = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'
    )

    # Build and compile the model
    model = build_model(input_shape=(img_size[0], img_size[1], 1))  # Ensure the model is built here

    # Train the model
    print("Starting training...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        verbose=1
    )

    # Save the model
    model.save(model_path)
    print(f"Model saved to {model_path}")

    return history

# Example usage
train_model("C:/Users/amith/signature_mini-project/data/processed_data", "signature_model.h5", img_size=(128, 128), batch_size=32, epochs=10)

