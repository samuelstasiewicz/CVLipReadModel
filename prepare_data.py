import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Configurable parameters
DATA_DIR = "data"
WORDS = ["goodbye", "hello", "hamburger"]
IMG_HEIGHT, IMG_WIDTH = 50, 100
FRAMES_PER_CLIP = 30

def load_data():
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
            for frame_name in frame_names[:FRAMES_PER_CLIP]:
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

    X = np.array(clips)
    X = np.expand_dims(X, axis=-1)  # (num_clips, 30, 50, 100, 1)
    y = np.array(labels)

    return train_test_split(X, y, test_size=0.2, random_state=42)


