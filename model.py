import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, TimeDistributed,
                                     Flatten, LSTM, Dense, Dropout)

# Set your parameters
DATA_DIR = "data"
WORDS = ["goodbye", "hello"]
IMG_HEIGHT, IMG_WIDTH = 50, 100
FRAMES_PER_CLIP = 30

# Prepare arrays
clips = []
labels = []

word_to_label = {word: idx for idx, word in enumerate(WORDS)}

for word in WORDS:
    word_path = os.path.join(DATA_DIR, word)
    clip_names = os.listdir(word_path)

    for clip_name in clip_names:
        clip_path = os.path.join(word_path, clip_name)
        frame_names = sorted(os.listdir(clip_path), key=lambda x: int(x.split('.')[0]))

        clip_frames = []
        for frame_name in frame_names[:FRAMES_PER_CLIP]:  # Make sure we get exactly 30 frames
            frame_path = os.path.join(clip_path, frame_name)
            img = cv2.imread(frame_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            img = img / 255.0  # normalize
            clip_frames.append(img)

        if len(clip_frames) == FRAMES_PER_CLIP:
            clip_frames = np.stack(clip_frames, axis=0)  # (30, 50, 100)
            clips.append(clip_frames)
            labels.append(word_to_label[word])

# Convert to numpy arrays
X = np.array(clips)  # (num_clips, 30, 50, 100)
X = np.expand_dims(X, axis=-1)  # (num_clips, 30, 50, 100, 1)
y = np.array(labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# Parameters
IMG_HEIGHT = 50
IMG_WIDTH = 100
FRAMES_PER_CLIP = 30
NUM_CLASSES = 2  # "hello" and "goodbye"

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