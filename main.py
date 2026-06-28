import os
import cv2
import face_recognition
import pickle

# -----------------------------
# Dataset Path
# -----------------------------
DATASET_PATH = "dataset"

known_face_encodings = []
known_face_names = []

print("Loading images...")

# -----------------------------
# Encode Dataset Images
# -----------------------------
for person_name in os.listdir(DATASET_PATH):

    person_folder = os.path.join(DATASET_PATH, person_name)

    if not os.path.isdir(person_folder):
        continue

    print(f"Processing {person_name}")

    for image_name in os.listdir(person_folder):

        image_path = os.path.join(person_folder, image_name)

        try:
            image = face_recognition.load_image_file(image_path)

            face_encodings = face_recognition.face_encodings(image)

            if len(face_encodings) > 0:
                known_face_encodings.append(face_encodings[0])
                known_face_names.append(person_name)
                print(f"   ✓ {image_name}")

            else:
                print(f"   ✗ No face found -> {image_name}")

        except Exception as e:
            print(f"Error reading {image_name}: {e}")

print("\nEncoding Completed!")

# -----------------------------
# Save Encodings
# -----------------------------
with open("encodings.pkl", "wb") as file:
    pickle.dump((known_face_encodings, known_face_names), file)

print("Encodings saved successfully!")

# -----------------------------
# Load Encodings
# -----------------------------
with open("encodings.pkl", "rb") as file:
    known_face_encodings, known_face_names = pickle.load(file)

print("\nStarting Webcam...")

# -----------------------------
# Start Webcam
# -----------------------------
video_capture = cv2.VideoCapture(0)

while True:

    ret, frame = video_capture.read()

    if not ret:
        break

    # Resize for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_small_frame)

    # Encode faces
    face_encodings = face_recognition.face_encodings(
        rgb_small_frame,
        face_locations
    )

    face_names = []

    for face_encoding in face_encodings:

        matches = face_recognition.compare_faces(
            known_face_encodings,
            face_encoding,
            tolerance=0.48
        )

        name = "Unknown"

        face_distances = face_recognition.face_distance(
            known_face_encodings,
            face_encoding
        )

        if len(face_distances) > 0:

            best_match_index = face_distances.argmin()

            if matches[best_match_index]:
                name = known_face_names[best_match_index]

        face_names.append(name)

    # Draw rectangles
    for (top, right, bottom, left), name in zip(face_locations, face_names):

        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame,
                      (left, top),
                      (right, bottom),
                      (0, 255, 0),
                      2)

        cv2.rectangle(frame,
                      (left, bottom - 35),
                      (right, bottom),
                      (0, 255, 0),
                      cv2.FILLED)

        cv2.putText(frame,
                    name,
                    (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 0),
                    2)

    cv2.imshow("Face Recognition using DLib", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()