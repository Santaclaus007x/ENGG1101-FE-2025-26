import cv2
import time
from ultralytics import YOLO

# COCO keypoint names (17 keypoints)
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# Load YOLOv26 nano pose model
print("Loading yolo26n-pose model...")
model = YOLO("yolo26n-pose.pt")

# Open webcam (Windows DirectShow backend)
cap = None
for idx in range(5):
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if cap.isOpened():
        print(f"Camera opened on index {idx}")
        break
    cap.release()

if not cap or not cap.isOpened():
    print("\n[ERROR] No camera found.")
    print("  Make sure your webcam is connected and not in use by another app.")
    exit(1)

# Set resolution (optional, adjust as needed)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Press 'q' to quit.")

wave_start_time = None

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Run pose estimation
    results = model(frame, device="cpu", verbose=False)

    # Draw the built-in annotated frame (skeleton + bounding boxes)
    annotated_frame = results[0].plot()

    # Default to no one waving
    any_person_waving = False

    # Overlay keypoint labels on each detected person
    if results[0].keypoints is not None:
        keypoints = results[0].keypoints.data  # shape: (num_people, 17, 3)

        for person_idx, person_kps in enumerate(keypoints):
            # Check for wave detection for this person
            # left_shoulder (5), right_shoulder (6), left_wrist (9), right_wrist (10)
            
            # Extract keypoints
            l_sh = person_kps[5]
            r_sh = person_kps[6]
            l_wr = person_kps[9]
            r_wr = person_kps[10]

            # Condition 1: Left wrist higher than left shoulder (smaller Y means higher in image space)
            if float(l_wr[2]) > 0.5 and float(l_sh[2]) > 0.5:
                if float(l_wr[1]) < float(l_sh[1]):
                    any_person_waving = True

            # Condition 2: Right wrist higher than right shoulder
            if float(r_wr[2]) > 0.5 and float(r_sh[2]) > 0.5:
                if float(r_wr[1]) < float(r_sh[1]):
                    any_person_waving = True

            for kp_idx, kp in enumerate(person_kps):
                x, y, conf = int(kp[0]), int(kp[1]), float(kp[2])

                if conf > 0.5:  # Only show keypoints with decent confidence
                    # Draw keypoint label
                    label = f"{KEYPOINT_NAMES[kp_idx]}"
                    cv2.putText(
                        annotated_frame, label, (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1,
                        cv2.LINE_AA
                    )

    if any_person_waving:
        if wave_start_time is None:
            wave_start_time = time.time()
        
        duration = time.time() - wave_start_time
        if duration >= 5.0:
            cv2.putText(
                annotated_frame, f"Wave Detected (5s+)!", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3,
                cv2.LINE_AA
            )
    else:
        wave_start_time = None

    cv2.imshow("YOLOv26 Pose Estimation", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
