import cv2
import mediapipe as mp
import os
import time

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Set which word you're recording
word = "hello"  # <-- CHANGE THIS EACH TIME ("yes", "no", "hello")
save_dir = f"data/{word}"

# Create folder if it doesn't exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Find the current highest clip number
existing_clips = [d for d in os.listdir(save_dir) if d.startswith('clip_')]
if existing_clips:
    existing_indices = [int(name.split('_')[1]) for name in existing_clips if name.split('_')[1].isdigit()]
    clip_count = max(existing_indices) + 1
else:
    clip_count = 0

print(f"Starting recording at clip number {clip_count}")

cap = cv2.VideoCapture(0)

recording = False
frames = []
frames_per_clip = 30  # how many frames per clip (about 0.5 seconds at 30fps)

print("Press SPACE to start recording a clip. Press ESC to quit.")

while True:
    success, frame = cap.read()
    if not success:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mouth_indices = list(range(78, 88)) + list(range(308, 318))  # inner mouth only
            h, w, _ = frame.shape
            mouth_points = [(int(landmark.x * w), int(landmark.y * h)) for idx, landmark in enumerate(face_landmarks.landmark) if idx in mouth_indices]

            if mouth_points:
                x_coords, y_coords = zip(*mouth_points)
                min_x, max_x = min(x_coords), max(x_coords)
                min_y, max_y = min(y_coords), max(y_coords)

                padding = 5
                min_x = max(0, min_x - padding)
                max_x = min(w, max_x + padding)
                min_y = max(0, min_y - padding)
                max_y = min(h, max_y + padding)

                mouth_crop = frame[min_y:max_y, min_x:max_x]

                if mouth_crop.size != 0:
                    mouth_crop = cv2.resize(mouth_crop, (100, 50))

                    if recording:
                        frames.append(mouth_crop)

                    cv2.imshow("Mouth", mouth_crop)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC to quit
        break
    elif key == 32:  # SPACE to start recording
        print("Recording clip...")
        recording = True
        frames = []

    if recording and len(frames) == frames_per_clip:
        clip_folder = os.path.join(save_dir, f"clip_{clip_count}")
        os.makedirs(clip_folder, exist_ok=True)
        for i, mouth_frame in enumerate(frames):
            cv2.imwrite(os.path.join(clip_folder, f"{i}.png"), mouth_frame)
        clip_count += 1
        print(f"Saved clip {clip_count}")
        recording = False
        frames = []

cap.release()
cv2.destroyAllWindows()

