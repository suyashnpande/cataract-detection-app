# Importing libraries for model training
import os
import json
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# Defining directory paths
base_dir = 'processed_images'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

# Initialize the ImageDataGenerator with data augmentation for training
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.10,
    height_shift_range=0.10,
    rescale=1/255,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # 20% of training data for validation
)

# Initialize the ImageDataGenerator for testing (without augmentation)
test_datagen = ImageDataGenerator(rescale=1./255)

# Define batch size
batch_size = 32

# Create training and validation data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='binary',
    subset='training'  # Set for training data
)

val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'  # Set for validation data
)

# Create testing data generator
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False  
)

# Input shape since VGG16 is used
input_shape = (224, 224, 3)

# Load the base model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

# Freeze all layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers on top of the base model for the classification task of cataract images
model = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation='relu', kernel_regularizer='l2'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Print the model summary for reference
model.summary()

# Compile the model with an initial learning rate
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model_v10.keras', monitor='val_loss', save_best_only=True)

# Train the model with frozen base layers
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator,
    callbacks=[early_stopping, model_checkpoint]
)

# Save training history
with open('training_history.json', 'w') as f:
    json.dump(history.history, f)

# Load the best model from training
model.load_weights('best_model_v10.keras')

# Fine-tune the model: Unfreeze the last few layers of the base model
for layer in base_model.layers[-4:]:
    layer.trainable = True

# Compile with a lower learning rate for fine-tuning
model.compile(optimizer=Adam(learning_rate=0.00001), loss='binary_crossentropy', metrics=['accuracy'])

# Continue training with fine-tuned layers
history_fine_tuning = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    callbacks=[early_stopping, model_checkpoint]
)

# Save final model
final_model_dir = os.path.join('..', 'cataract_app')
if not os.path.exists(final_model_dir):
    os.makedirs(final_model_dir)
final_model_path = os.path.join(final_model_dir, 'final_model.keras')
model.save(final_model_path)

# Save fine-tuning history
with open('fine_tuning_history.json', 'w') as f:
    json.dump(history_fine_tuning.history, f)

# Plotting the training and validation accuracy/loss
def plot_history(histories, keys):
    plt.figure(figsize=(12, 4))
    for i, key in enumerate(keys):
        plt.subplot(1, 2, i+1)
        for name, history in histories.items():
            plt.plot(history[key], label=name)
        plt.title(key)
        plt.xlabel('Epochs')
        plt.ylabel(key.capitalize())
        plt.legend()
    plt.show()

# Combine both training histories for better visualization
combined_history = {
    'initial_training': history.history,
    'fine_tuning': history_fine_tuning.history
}
plot_history(combined_history, ['accuracy', 'loss'])
