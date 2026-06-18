import cv2                          # OpenCV — webcam, drawing, display window
import time                         # FPS calculation
import numpy as np                  # Array operations for frame processing
from datetime import datetime       # Timestamp for saved screenshots
from ultralytics import YOLO        # YOLOv8 model loader and runner

CAMERA_INDEX    = 0             # 0 = built-in webcam. Change to 1 for USB camera.
FRAME_WIDTH     = 640           # Video frame width in pixels
FRAME_HEIGHT    = 480           # Video frame height in pixels
CONFIDENCE_MIN  = 0.40          # Minimum confidence to accept a detection (40%)
MODEL_PATH      = "yolov8n.pt"  # YOLOv8 nano — fastest model, ideal for CPU
OUTPUT_DIR      = "../outputs"  # Folder for saved screenshots

PERSON_CLASS_ID = 0

PERSON_COLOR = (0, 230, 0)

NUMBER_COLOR = (255, 255, 255)

def draw_person_box(frame, box, person_number: int, confidence: float) -> None:
  
    x1 = int(box.xyxy[0][0])
    y1 = int(box.xyxy[0][1])
    x2 = int(box.xyxy[0][2])
    y2 = int(box.xyxy[0][3])

    # Draw the green bounding box around the person
    # Thickness of 3 makes it clearly visible even on busy backgrounds
    cv2.rectangle(frame, (x1, y1), (x2, y2), PERSON_COLOR, 3)

    # Build label text: e.g. "#1  Person  93%"
    label = f"#{person_number}  Person  {confidence:.0%}"

    # Measure label size so we can draw a proper background behind it
    (text_w, text_h), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2
    )

    label_y = max(y1 - 2, text_h + 10)

    cv2.rectangle(
        frame,
        (x1, label_y - text_h - 8),
        (x1 + text_w + 8, label_y + 2),
        (20, 20, 20),       # Very dark background — visible over any scene
        cv2.FILLED
    )
  
    cv2.rectangle(
        frame,
        (x1, label_y - text_h - 8),
        (x1 + text_w + 8, label_y + 2),
        PERSON_COLOR,
        1
    )

    # Write the label text in white
    cv2.putText(
        frame, label,
        (x1 + 4, label_y - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        NUMBER_COLOR,
        2,
        cv2.LINE_AA
    )


# ── DRAW PERSON COUNT PANEL ───────────────────────────────────────────────────
def draw_count_panel(frame, person_count: int, fps: float) -> None:
    """
    The count changes color based on how many people are detected:
    - Green  (0–2 persons) : Normal, safe level
    - Yellow (3–4 persons) : Getting crowded — approaching alert threshold
    - Red    (5+ persons)  : High crowd — would trigger Phase 3 alert

    This color coding gives a visual warning even before the audio alert in Phase 3.

    Args:
        frame        : Current video frame
        person_count : Total persons detected in this frame
        fps          : Current frames per second
    """
    frame_w = frame.shape[1]
    frame_h = frame.shape[0]

    # ── Top information bar ───────────────────────────────────────────────────
    # Draw semi-transparent dark bar across the full top of the frame
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame_w, 50), (15, 15, 15), cv2.FILLED)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # FPS — top left in green
    cv2.putText(frame, f"FPS: {fps:.1f}",
                (12, 33), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 220, 100), 2, cv2.LINE_AA)

    # Phase label — top right in soft blue
    phase_text = "Phase 2: Person Detection"
    (pw, _), _ = cv2.getTextSize(phase_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.putText(frame, phase_text,
                (frame_w - pw - 12, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 200, 255), 1, cv2.LINE_AA)

    if person_count == 0:
        count_color = (160, 160, 160)    # Gray — nobody detected
    elif person_count <= 2:
        count_color = (0, 220, 80)       # Green — safe, within limit
    elif person_count <= 4:
        count_color = (0, 200, 255)      # Yellow-orange — getting crowded
    else:
        count_color = (0, 60, 255)       # Red — crowd detected (Phase 3 alert zone)

    panel_x, panel_y = 10, frame_h - 100
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (panel_x, panel_y),
                  (panel_x + 210, panel_y + 85), (15, 15, 15), cv2.FILLED)
    cv2.addWeighted(overlay2, 0.8, frame, 0.2, 0, frame)

    # Colored border around the count panel — matches the count color
    cv2.rectangle(frame, (panel_x, panel_y),
                  (panel_x + 210, panel_y + 85), count_color, 2)

    # Small label "PERSONS DETECTED" above the number
    cv2.putText(frame, "PERSONS DETECTED",
                (panel_x + 8, panel_y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (180, 180, 180), 1, cv2.LINE_AA)

    # Large person count number — main focus of this display
    count_str = str(person_count)
    (cw, ch), _ = cv2.getTextSize(count_str, cv2.FONT_HERSHEY_SIMPLEX, 2.2, 4)
    count_x = panel_x + (210 - cw) // 2     # Center the number in the panel
    cv2.putText(frame, count_str,
                (count_x, panel_y + 72),
                cv2.FONT_HERSHEY_SIMPLEX, 2.2,
                count_color, 4, cv2.LINE_AA)

    # ── Bottom keyboard hints bar ─────────────────────────────────────────────
    hint_overlay = frame.copy()
    cv2.rectangle(hint_overlay, (0, frame_h - 30),
                  (frame_w, frame_h), (15, 15, 15), cv2.FILLED)
    cv2.addWeighted(hint_overlay, 0.75, frame, 0.25, 0, frame)
    cv2.putText(frame, "Press 'S' to save screenshot  |  Press 'Q' to quit",
                (12, frame_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (180, 180, 180), 1, cv2.LINE_AA)

def main():
    print("=" * 60)
    print("  Phase 2: Person-Only Detection")
    print("  Filtering: class_id == 0 (person) only")
    print("  All other objects are ignored")
    print("  Camera: Index", CAMERA_INDEX)
    print("=" * 60)

    print("\n[INFO] Loading YOLOv8n model...")
    model = YOLO(MODEL_PATH)
    print("[INFO] Model loaded. Ready for person detection.")
    print(f"[INFO] Person class ID in COCO dataset = {PERSON_CLASS_ID}")

    # ── Open webcam ───────────────────────────────────────────────────────────
    print(f"\n[INFO] Opening camera (index {CAMERA_INDEX})...")
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        print("        Try changing CAMERA_INDEX to 1 at the top of this file.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    ret, test_frame = cap.read()
    if ret:
        h, w = test_frame.shape[:2]
        print(f"[INFO] Camera ready. Resolution: {w}x{h}")
    else:
        print("[ERROR] Could not read from camera.")
        cap.release()
        return

    print("\n[INFO] Person detection running.")
    print("[INFO] Move people in front of the camera to see them detected.")
    print("[INFO] Press 'Q' to quit, 'S' to save screenshot.")
    print("-" * 60)

    # ── FPS tracking variables ────────────────────────────────────────────────
    fps_start   = time.time()
    fps_counter = 0
    fps_display = 0.0

    # ── Main detection loop ───────────────────────────────────────────────────
    while True:

        # Read one frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Lost camera connection.")
            break

        results    = model(frame, conf=CONFIDENCE_MIN, verbose=False)
        detections = results[0].boxes

        person_count  = 0     # Resets to 0 for every new frame
        person_number = 0     # Used to number each person box (#1, #2, #3...)

        if detections is not None:
            for box in detections:

                # Read what class this detection is
                class_id = int(box.cls[0])

        
             
                if class_id != PERSON_CLASS_ID:
                    continue
       

                confidence = float(box.conf[0])
                person_count  += 1       # Increment total person count
                person_number += 1       # Increment per-box numbering

                # Draw green box and label for this person
                draw_person_box(frame, box, person_number, confidence)

 
        fps_counter += 1
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            fps_display = fps_counter / elapsed
            fps_counter = 0
            fps_start   = time.time()

        draw_count_panel(frame, person_count, fps_display)

        cv2.imshow("Phase 2 — Person Detection | Techlive Solutions", frame)
      
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == ord('Q'):
            print(f"\n[INFO] Quit. Last frame had {person_count} person(s) detected.")
            break

        elif key == ord('s') or key == ord('S'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename  = f"{OUTPUT_DIR}/phase2_persons_{person_count}detected_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"[INFO] Screenshot saved → {filename}")
            print(f"[INFO] Persons in screenshot: {person_count}")

    print("\n[INFO] Releasing camera and closing display window...")
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Phase 2 complete.")
    print("=" * 60)

if __name__ == "__main__":
    main()
