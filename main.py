import cv2
import mediapipe as mp
import numpy as np
import time
import winsound  # Specific for Windows alarm sound

# --- CONFIGURATION ---
EAR_THRESHOLD = 0.26    # Eye Aspect Ratio threshold to indicate closed eyes
SLEEP_FRAMES = 75       # Number of consecutive frames to trigger alarm (approx 2.5s)
CAM_WIDTH, CAM_HEIGHT = 640, 480

# --- MEDIAPIPE SETUP ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# --- EYE LANDMARK INDICES (MediaPipe) ---
# Specific landmark IDs for Left and Right eyes
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# --- FUNCTION: Calculate Eye Aspect Ratio (EAR) ---
def calculate_ear(landmarks, indices, w, h):
    """
    Calculates the Eye Aspect Ratio (EAR) to detect if the eye is open or closed.
    EAR = (Distance_Vertical_1 + Distance_Vertical_2) / (2 * Distance_Horizontal)
    """
    coords = []
    for i in indices:
        lm = landmarks[i]
        coords.append([int(lm.x * w), int(lm.y * h)])
    
    # Vertical distances (Eye Lid Openness)
    d_A = np.linalg.norm(np.array(coords[1]) - np.array(coords[5]))
    d_B = np.linalg.norm(np.array(coords[2]) - np.array(coords[4]))

    # Horizontal distance (Eye Width)
    d_C = np.linalg.norm(np.array(coords[0]) - np.array(coords[3]))

    # EAR Formula
    ear = (d_A + d_B) / (2.0 * d_C)
    return ear, coords

# --- MAIN LOOP ---
# Using CAP_DSHOW for faster initialization on Windows/Monster Notebooks
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 
cap.set(3, CAM_WIDTH)
cap.set(4, CAM_HEIGHT)

frame_counter = 0  
alarm_on = False

print("\n[INFO] System Starting... Monitoring driver attention! 👀")
print("[INFO] Press 'q' to exit.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("[WARNING] Frame lost, skipping...")
        continue

    # Convert BGR to RGB for MediaPipe processing
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    # Convert back to BGR for rendering
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    h, w, c = image.shape

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            # Calculate EAR for both eyes
            left_ear, left_coords = calculate_ear(landmarks, LEFT_EYE, w, h)
            right_ear, right_coords = calculate_ear(landmarks, RIGHT_EYE, w, h)

            # Average EAR of both eyes
            avg_ear = (left_ear + right_ear) / 2.0

            # Visualization Color (Green = Safe, Red = Danger)
            color = (0, 255, 0) 
            
            # --- DROWSINESS LOGIC ---
            if avg_ear < EAR_THRESHOLD:
                frame_counter += 1
                color = (0, 0, 255) # Red

                # Trigger Alarm if eyes are closed for sufficient time
                if frame_counter >= SLEEP_FRAMES:
                    alarm_on = True
                    cv2.putText(image, "WARNING: DROWSINESS DETECTED!", (20, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                    
                    # Play Beep Sound (Frequency: 2500Hz, Duration: 100ms)
                    # Non-blocking beep logic could be implemented with threading if needed
                    winsound.Beep(2500, 100)
            else:
                frame_counter = 0
                alarm_on = False
                color = (0, 255, 0)

            # Draw Eye Contours
            cv2.polylines(image, [np.array(left_coords)], True, color, 1)
            cv2.polylines(image, [np.array(right_coords)], True, color, 1)

            # Display Stats on Screen
            cv2.putText(image, f"EAR: {avg_ear:.2f}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if alarm_on:
                 cv2.putText(image, "ALARM ACTIVE!", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Show the Output
    cv2.imshow('Driver Drowsiness Detection System', image)
    
    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()