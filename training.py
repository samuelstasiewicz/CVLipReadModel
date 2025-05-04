import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, TimeDistributed, Flatten, LSTM, Dense, Dropout)
from prepare_data import load_data
from prepare_data import WORDS

X_train, X_test, y_train, y_test = load_data()

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")


# Parameters
IMG_HEIGHT = 50
IMG_WIDTH = 100
FRAMES_PER_CLIP = 30
NUM_CLASSES = len(WORDS)  # "hello" and "goodbye"

# Build the model
model = Sequential()

# TimeDistributed CNN
model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu'),
                           input_shape=(FRAMES_PER_CLIP, IMG_HEIGHT, IMG_WIDTH, 1)))
model.add(TimeDistributed(MaxPooling2D((2, 2))))
model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu')))
model.add(TimeDistributed(MaxPooling2D((2, 2))))
model.add(TimeDistributed(Flatten()))

# LSTM for temporal patterns
model.add(LSTM(128, return_sequences=False))
model.add(Dropout(0.5))

# Dense output
model.add(Dense(NUM_CLASSES, activation='softmax'))

# Compile
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Show model summary
model.summary()

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=5,           # Number of passes over the data
    batch_size=8,        # How many clips to process at a time
    verbose=1            # Shows training progress
)

import matplotlib.pyplot as plt

# Plot training & validation accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.show()