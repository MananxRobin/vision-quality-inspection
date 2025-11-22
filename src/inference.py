import cv2
import numpy as np
import onnxruntime as ort

# --- CONFIGURATION ---
MODEL_PATH = "../models/defect_detector.onnx"
CONFIDENCE_THRESHOLD = 0.7  # If confidence is lower, we can say "Uncertain"
# ---------------------

# 1. Load the optimized ONNX model
print("Loading model...")
ort_session = ort.InferenceSession(MODEL_PATH)
print("Model loaded!")


def preprocess_image(frame):
    # A. Resize to match training size (256x256)
    img = cv2.resize(frame, (256, 256))

    # B. Convert Color: OpenCV uses BGR, Model expects RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # C. Normalize (Subtract Mean, Divide Std - Standard ImageNet stats)
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std

    # D. Transpose to (Channels, Height, Width)
    img = img.transpose(2, 0, 1)

    # E. Add Batch Dimension (1, 3, 256, 256)
    img = np.expand_dims(img, axis=0)

    return img.astype(np.float32)


# 2. Start Camera (0 is usually the default webcam)
cap = cv2.VideoCapture(0)

# Optional: Use a video file if you don't have a webcam
# cap = cv2.VideoCapture("test_video.mp4")

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Starting Inference... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Save original frame for display
    display_frame = frame.copy()

    # 3. Preprocess
    input_tensor = preprocess_image(frame)

    # 4. Run Inference (ONNX)
    # "input" and "output" must match the names used in export_onnx.py
    outputs = ort_session.run(["output"], {"input": input_tensor})
    logits = outputs[0][0]

    # 5. Post-process (Softmax)
    probs = np.exp(logits) / np.sum(np.exp(logits))
    predicted_idx = np.argmax(probs)
    confidence = probs[predicted_idx]

    # Classes are alphabetical: 0=defective, 1=good
    # (Double check your train.py 'Classes found' output to confirm order!)
    class_names = ['DEFECTIVE', 'GOOD']
    label = class_names[predicted_idx]

    # 6. UI Visualization
    color = (0, 255, 0) if label == 'GOOD' else (0, 0, 255)  # Green vs Red

    # Dynamic Thresholding (Resume point: "Reduced escape rates")
    if confidence < CONFIDENCE_THRESHOLD:
        label = "UNCERTAIN"
        color = (0, 255, 255)  # Yellow

    # Draw Box and Text
    height, width, _ = display_frame.shape
    cv2.rectangle(display_frame, (0, 0), (width, 60), (0, 0, 0), -1)  # Top bar
    cv2.putText(display_frame, f"Result: {label}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(display_frame, f"Conf: {confidence:.2%}", (width - 200, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Quality Inspection System", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()