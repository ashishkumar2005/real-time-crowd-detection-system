import os
import cv2
import time
import numpy as np
from datetime import datetime
from ultralytics import YOLO

# CONFIGURATION
# Edit these values if needed — everything else in the code adjusts automatically
CAMERA_INDEX   = 0          # 0 = default built-in webcam. Change to 1 for USB camera.
FRAME_WIDTH    = 640        # Width of the video frame in pixels
FRAME_HEIGHT   = 480        # Height of the video frame in pixels
CONFIDENCE_MIN = 0.40       # Only show detections above 40% confidence (0.0 to 1.0)
WINDOW_NAME    = "Phase 1 - Object Detection | Techlive Solutions"


MODEL_PATH = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "yolov8n.pt")
)

# Absolute path to outputs folder for saving screenshots
OUTPUT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
)


# FUNCTION: generate_colors
def generate_colors(num_classes: int) -> list:
    """
    Generate a unique BGR color for each YOLO object class.

    YOLOv8 detects 80 object types. We assign each type a unique color
    so bounding boxes are visually easy to tell apart.

    Uses HSV color space for even color distribution across the spectrum,
    then converts to BGR (the format OpenCV uses for drawing).

    Args:
        num_classes: Total number of object classes (80 for COCO dataset)

    Returns:
        List of (B, G, R) tuples — one color per class
    """
    colors = []
    for i in range(num_classes):
        # Spread hue evenly across 0 to 179 (OpenCV HSV hue range)
        hue = int(179 * i / num_classes)

        hsv_array = np.array([[[hue, 220, 200]]], dtype=np.uint8)
        bgr_array = cv2.cvtColor(hsv_array, cv2.COLOR_HSV2BGR)
        hsv_pixel = bgr_array[0][0]

        # Convert numpy uint8 values to plain Python ints for OpenCV drawing
        colors.append((int(hsv_pixel[0]), int(hsv_pixel[1]), int(hsv_pixel[2])))

    return colors


# FUNCTION: draw_detection
def draw_detection(frame, box, class_id: int, confidence: float,
                   class_name: str, color: tuple) -> None:
    """
    Draw a single detection's bounding box and label onto the video frame.

    Args:
        frame      : The current video frame (NumPy array from OpenCV)
        box        : YOLO bounding box — contains x1, y1, x2, y2 coordinates
        class_id   : Integer ID of the detected class (0 to 79 for COCO)
        confidence : Float between 0 and 1 — how sure YOLO is about this detection
        class_name : Human-readable name like "person", "bottle", "laptop"
        color      : (B, G, R) tuple for this object class
    """
    # Extract pixel coordinates of the bounding box corners
    x1 = int(box.xyxy[0][0])
    y1 = int(box.xyxy[0][1])
    x2 = int(box.xyxy[0][2])
    y2 = int(box.xyxy[0][3])

    # Draw the bounding box rectangle on the frame
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Build the label text: e.g. "laptop  87%"
    label = f"{class_name}  {confidence:.0%}"

    # Measure how big the label text will be so we can draw a background behind it
    (text_w, text_h), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
    )

    # Keep label inside frame if box is at the very top
    label_y = max(y1, text_h + 10)

    # Draw filled rectangle as background behind the label so text is readable
    cv2.rectangle(
        frame,
        (x1, label_y - text_h - 8),
        (x1 + text_w + 6, label_y),
        color,
        cv2.FILLED
    )

    # Choose text color — white on dark boxes, black on bright boxes
    brightness = 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]
    text_color = (0, 0, 0) if brightness > 140 else (255, 255, 255)

    # Write the label text over the colored background
    cv2.putText(
        frame, label,
        (x1 + 3, label_y - 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        text_color,
        1,
        cv2.LINE_AA
    )


# FUNCTION: draw_hud
def draw_hud(frame, fps: float, total_detections: int) -> None:
    """
    Draw the Heads-Up Display overlay at the top and bottom of the frame.

    Top bar shows: FPS (left), Object count (center), Phase label (right)
    Bottom bar shows: Keyboard controls hint

    Args:
        frame            : Current video frame
        fps              : Calculated frames per second
        total_detections : Number of objects detected in this frame
    """
    frame_w = frame.shape[1]
    frame_h = frame.shape[0]

    # Semi-transparent dark bar at the top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame_w, 50), (20, 20, 20), cv2.FILLED)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # FPS counter — top left in green
    cv2.putText(frame, f"FPS: {fps:.1f}",
                (12, 33), cv2.FONT_HERSHEY_SIMPLEX,
                0.80, (0, 255, 100), 2, cv2.LINE_AA)

    # Object count — top center in yellow
    count_text = f"Objects: {total_detections}"
    (tw, _), _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.80, 2)
    cv2.putText(frame, count_text,
                (frame_w // 2 - tw // 2, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.80,
                (0, 220, 255), 2, cv2.LINE_AA)

    # Phase label — top right in soft blue
    proj_text = "Phase 1: Object Detection"
    (pw, _), _ = cv2.getTextSize(proj_text, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
    cv2.putText(frame, proj_text,
                (frame_w - pw - 20, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                (180, 180, 255), 1, cv2.LINE_AA)

    # Semi-transparent dark bar at the bottom
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, frame_h - 34), (frame_w, frame_h),
                  (20, 20, 20), cv2.FILLED)
    cv2.addWeighted(overlay2, 0.75, frame, 0.25, 0, frame)

    # Keyboard hint text
    cv2.putText(frame,
                "Press 'S' to save screenshot  |  Press 'Q' to quit",
                (12, frame_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                (200, 200, 200), 1, cv2.LINE_AA)


# FUNCTION: main
def main():
   
    print("=" * 60)
    print("  Phase 1: Real-Time Object Detection")
    print("  Model  : YOLOv8n (COCO - 80 classes)")
    print(f"  Model path : {MODEL_PATH}")
    print(f"  Output dir : {OUTPUT_DIR}")
    print("=" * 60)

    # Create outputs folder if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # STEP 1: Load the YOLOv8 model
    # "yolov8n.pt" downloads automatically (~6 MB) on first run.
    # After that it loads from the models\ folder instantly.
    print("\n[INFO] Loading YOLOv8n model...")
    print(f"[INFO] Looking for model at: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print("[INFO] Model loaded successfully.")

    # Get the list of 80 class names (person, bicycle, car, laptop, chair, etc.)
    class_names = model.names
    print(f"[INFO] Model can detect {len(class_names)} object types.")

    # Generate a unique color for each of the 80 object classes
    colors = generate_colors(len(class_names))

    # STEP 2: Open the webcam
    print(f"\n[INFO] Opening camera (index {CAMERA_INDEX})...")
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("[ERROR] Cannot open camera. Check if webcam is connected.")
        print("        Try changing CAMERA_INDEX from 0 to 1 at the top of this file.")
        return

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    # Test read to confirm camera works
    ret, test_frame = cap.read()
    if ret:
        actual_h, actual_w = test_frame.shape[:2]
        print(f"[INFO] Camera opened. Resolution: {actual_w}x{actual_h}")
    else:
        print("[ERROR] Could not read from camera.")
        cap.release()
        return

    # STEP 3: Create a resizable window with custom size
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    # Set the display window size (change these values as needed)
    cv2.resizeWindow(WINDOW_NAME, 1366, 768)

    print("\n[INFO] Detection started in fullscreen mode.")
    print("[INFO] Press 'Q' to quit, 'S' to save screenshot.")
    print("-" * 60)

    # FPS tracking variables
    # fps_start   : timestamp when FPS measurement period began
    # fps_counter : how many frames processed in current measurement period
    # fps_display : the FPS value shown on screen (updates every 1 second)
    fps_start   = time.time()
    fps_counter = 0
    fps_display = 0.0

    # STEP 4: Main detection loop
    # Runs forever until Q is pressed.
    # Each loop iteration processes exactly one video frame.
    while True:

        # Read the next frame from webcam
        # ret   : True if frame read successfully
        # frame : NumPy array of shape (480, 640, 3) — height x width x BGR
        ret, frame = cap.read()

        if not ret:
            print("[WARNING] Failed to read frame. Camera may have disconnected.")
            break

        # STEP 5: Run YOLO detection on this frame
        # model(frame) sends the frame through the neural network.
        # conf=CONFIDENCE_MIN filters out low-confidence detections.
        # verbose=False stops YOLO from printing results to terminal every frame.
        results    = model(frame, conf=CONFIDENCE_MIN, verbose=False)
        detections = results[0].boxes

        # STEP 6: Draw each detection on the frame
        total_count = 0

        if detections is not None:
            for box in detections:
                class_id   = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = class_names[class_id]
                color      = colors[class_id]

                draw_detection(frame, box, class_id, confidence, class_name, color)
                total_count += 1

        # STEP 7: Calculate FPS
        # Count frames for 1 second, then calculate how many fit in that second
        fps_counter += 1
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            fps_display = fps_counter / elapsed
            fps_counter = 0
            fps_start   = time.time()

        # STEP 8: Draw HUD overlay on the frame
        draw_hud(frame, fps_display, total_count)

        # STEP 9: Show the annotated frame in the fullscreen window
        cv2.imshow(WINDOW_NAME, frame)

        # STEP 10: Check for keyboard input
        # waitKey(1) waits 1ms for a key. 0xFF mask gets the ASCII code.
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == ord('Q'):
            print("\n[INFO] Q pressed. Stopping detection...")
            break

        elif key == ord('s') or key == ord('S'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename  = os.path.join(OUTPUT_DIR, f"phase1_screenshot_{timestamp}.jpg")
            cv2.imwrite(filename, frame)
            print(f"[INFO] Screenshot saved: {filename}")

    # STEP 11: Cleanup
    # Always release camera and destroy windows.
    # Not doing this leaves the webcam locked for other apps.
    print("[INFO] Releasing camera and closing window...")
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Phase 1 complete.")
    print("=" * 60)

# ENTRY POINT
# This block runs only when you execute this file directly.
# It does NOT run if another script imports this file as a module.
if __name__ == "__main__":
    main()
