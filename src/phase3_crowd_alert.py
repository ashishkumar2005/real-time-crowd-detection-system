import cv2                      # OpenCV — webcam, drawing, display window
import time                     # FPS calculation and alert cooldown timer
import csv                      # Writing alert events to CSV log file
import os                       # Folder creation and path handling
import threading                # Run audio beep in background (non-blocking)
import numpy as np              # Array operations for frame/color processing
from datetime import datetime   # Timestamps for logs and saved filenames
from ultralytics import YOLO    # YOLOv8 detection model

# ── AUDIO — Windows winsound ──────────────────────────────────────────────────
try:
    import winsound
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

OUTPUT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
)

# ── ALERT SETTINGS ────────────────────────────────────────────────────────────
CROWD_THRESHOLD = 2      # Alert fires when person count EXCEEDS this number
                         # Value 2 means: 3 or more persons → alert fires

ALERT_COOLDOWN  = 3.0   # Seconds between consecutive beep alerts
                         # Prevents speaker from being hammered every frame

BEEP_FREQUENCY  = 1000  # Beep pitch in Hz (1000 = clear audible tone)
BEEP_DURATION   = 500   # Beep length in milliseconds (500 = half a second)

# ── COLORS (BGR format — OpenCV uses Blue,Green,Red not Red,Green,Blue) ───────
COLOR_PERSON_SAFE   = (0, 230, 0)      # Green  — normal person box
COLOR_PERSON_ALERT  = (0, 60, 255)     # Red    — person box during alert
COLOR_COUNT_SAFE    = (0, 220, 80)     # Green  — count panel safe state
COLOR_COUNT_DANGER  = (0, 60, 255)     # Red    — count panel alert state
COLOR_ALERT_BG      = (0, 0, 180)      # Dark red — alert banner background

PERSON_CLASS_ID = 0   # "person" is always class 0 in COCO dataset


# ── STEP 1: CREATE OUTPUT FOLDER ──────────────────────────────────────────────
def setup_output_dir(path: str) -> bool:
    """
    Create outputs folder and verify we can actually write to it.
    Returns True if folder is ready, False if there is a permission problem.

    We test write access by creating and immediately deleting a tiny test file.
    This catches permission errors BEFORE the program starts detecting —
    so we know early rather than crashing mid-detection.
    """
    try:
        os.makedirs(path, exist_ok=True)

        # Test that we can actually write a file here
        test_file = os.path.join(path, "_write_test.tmp")
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)

        print(f"[INFO] Output folder ready: {path}")
        return True

    except PermissionError:
        print(f"[ERROR] Cannot write to output folder: {path}")
        print(f"[ERROR] Fix: Right-click the outputs folder → Properties → Uncheck Read-only")
        return False
    except Exception as e:
        print(f"[ERROR] Output folder problem: {e}")
        return False


# ── STEP 2: PLAY BEEP IN BACKGROUND THREAD ────────────────────────────────────
def play_beep_async(frequency: int, duration: int) -> None:
    """
    Play alert beep in a separate background thread.

    WHY THREADING?
    winsound.Beep() is a blocking call — it makes the program WAIT until
    the beep finishes before doing anything else. At 500ms beep duration,
    that means the camera feed freezes for half a second on every alert.
    With threading, the beep plays in the background while the camera
    continues reading and displaying frames normally — no freezing.

    threading.Thread(target=func) creates a new thread.
    daemon=True means the thread dies automatically when main program exits.
    .start() launches it immediately without waiting for it to finish.
    """
    if not AUDIO_AVAILABLE:
        return

    def _beep():
        try:
            winsound.Beep(frequency, duration)
        except Exception as e:
            print(f"[WARNING] Beep error: {e}")

    t = threading.Thread(target=_beep, daemon=True)
    t.start()


# ── STEP 3: SAVE SCREENSHOT ────────────────────────────────────────────────────
def save_screenshot(frame, output_dir: str, person_count: int) -> str:
    """
    Save the current frame as a JPEG screenshot.
    Returns the full file path of the saved image (used in CSV log).
    Returns empty string if save fails.
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename  = f"CROWD_ALERT_{person_count}persons_{timestamp}.jpg"
        filepath  = os.path.join(output_dir, filename)
        success   = cv2.imwrite(filepath, frame)
        if success:
            print(f"[INFO] Screenshot saved → {filepath}")
            return filepath
        else:
            print(f"[WARNING] Screenshot save failed (cv2.imwrite returned False)")
            return ""
    except Exception as e:
        print(f"[WARNING] Screenshot error: {e}")
        return ""


# ── STEP 4: LOG TO CSV ────────────────────────────────────────────────────────
def log_to_csv(output_dir: str, person_count: int, screenshot_path: str) -> None:
    """
    Append one alert event to crowd_log.csv.

    CSV columns: timestamp, persons_detected, alert_status, screenshot_file

    Uses try/except so a CSV write failure never crashes the whole program.
    The camera keeps running even if logging fails.
    """
    log_file   = os.path.join(output_dir, "crowd_log.csv")
    file_exists = os.path.isfile(log_file)

    try:
        with open(log_file, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Write header row only on first entry
            if not file_exists:
                writer.writerow([
                    "timestamp",
                    "persons_detected",
                    "alert_status",
                    "screenshot_file"
                ])

            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                person_count,
                "CROWD ALERT",
                os.path.basename(screenshot_path) if screenshot_path else "none"
            ])

        print(f"[INFO] Event logged to crowd_log.csv")

    except PermissionError:
        print(f"[WARNING] Cannot write to CSV — file may be open in Excel.")
        print(f"[WARNING] Close crowd_log.csv in Excel then it will log again.")
    except Exception as e:
        print(f"[WARNING] CSV log error: {e}")


# ── DRAW PERSON BOX ───────────────────────────────────────────────────────────
def draw_person_box(frame, box, person_number: int,
                    confidence: float, alert_active: bool) -> None:
    """
    Draw bounding box and label for one detected person.
    Box color: green normally, red when alert is active.
    """
    x1 = int(box.xyxy[0][0])
    y1 = int(box.xyxy[0][1])
    x2 = int(box.xyxy[0][2])
    y2 = int(box.xyxy[0][3])

    box_color = COLOR_PERSON_ALERT if alert_active else COLOR_PERSON_SAFE

    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)

    label = f"#{person_number}  Person  {confidence:.0%}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)

    label_y = max(y1 - 2, th + 10)

    cv2.rectangle(frame,
                  (x1, label_y - th - 8),
                  (x1 + tw + 8, label_y + 2),
                  (15, 15, 15), cv2.FILLED)

    cv2.rectangle(frame,
                  (x1, label_y - th - 8),
                  (x1 + tw + 8, label_y + 2),
                  box_color, 1)

    cv2.putText(frame, label, (x1 + 4, label_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                (255, 255, 255), 2, cv2.LINE_AA)


# ── DRAW ALERT BANNER ─────────────────────────────────────────────────────────
def draw_alert_banner(frame, person_count: int, blink_on: bool) -> None:
    """
    Draw blinking red warning banner in the center of the frame.
    blink_on alternates True/False every 15 frames to create pulse effect.
    """
    fh, fw = frame.shape[:2]
    y1 = fh // 2 - 55
    y2 = fh // 2 + 55

    opacity = 0.88 if blink_on else 0.40
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y1), (fw, y2), COLOR_ALERT_BG, cv2.FILLED)
    cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)

    cv2.line(frame, (0, y1), (fw, y1), (255, 255, 255), 2)
    cv2.line(frame, (0, y2), (fw, y2), (255, 255, 255), 2)

    line1 = "!! CROWD ALERT !!"
    (w1, _), _ = cv2.getTextSize(line1, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
    cv2.putText(frame, line1,
                (fw // 2 - w1 // 2, fh // 2 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                (255, 255, 255), 3, cv2.LINE_AA)

    line2 = f"{person_count} PERSONS DETECTED  —  THRESHOLD EXCEEDED"
    (w2, _), _ = cv2.getTextSize(line2, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.putText(frame, line2,
                (fw // 2 - w2 // 2, fh // 2 + 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 2, cv2.LINE_AA)


# ── DRAW HUD ──────────────────────────────────────────────────────────────────
def draw_hud(frame, person_count: int, fps: float,
             alert_active: bool, total_alerts: int) -> None:
    """
    Draw top information bar, bottom-left count panel, bottom hint bar.
    """
    fh, fw = frame.shape[:2]
    count_color = COLOR_COUNT_DANGER if alert_active else COLOR_COUNT_SAFE

    # Top bar
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (fw, 50), (15, 15, 15), cv2.FILLED)
    cv2.addWeighted(ov, 0.78, frame, 0.22, 0, frame)

    cv2.putText(frame, f"FPS: {fps:.1f}",
                (12, 33), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 220, 100), 2, cv2.LINE_AA)

    alert_str = f"Alerts fired: {total_alerts}"
    (aw, _), _ = cv2.getTextSize(alert_str, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.putText(frame, alert_str,
                (fw // 2 - aw // 2, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 60, 255) if alert_active else (160, 160, 160),
                1, cv2.LINE_AA)

    phase_str = "Phase 3: Crowd Alert System"
    (pw, _), _ = cv2.getTextSize(phase_str, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
    cv2.putText(frame, phase_str,
                (fw - pw - 12, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (180, 200, 255), 1, cv2.LINE_AA)

    # Count panel — bottom left
    px, py = 10, fh - 108
    ov2 = frame.copy()
    cv2.rectangle(ov2, (px, py), (px + 215, py + 93), (15, 15, 15), cv2.FILLED)
    cv2.addWeighted(ov2, 0.82, frame, 0.18, 0, frame)
    cv2.rectangle(frame, (px, py), (px + 215, py + 93), count_color, 2)

    cv2.putText(frame, "PERSONS DETECTED",
                (px + 8, py + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (180, 180, 180), 1, cv2.LINE_AA)

    cstr = str(person_count)
    (cw, _), _ = cv2.getTextSize(cstr, cv2.FONT_HERSHEY_SIMPLEX, 2.4, 4)
    cv2.putText(frame, cstr,
                (px + (215 - cw) // 2, py + 78),
                cv2.FONT_HERSHEY_SIMPLEX, 2.4,
                count_color, 4, cv2.LINE_AA)

    status_str = "ALERT ACTIVE" if alert_active else f"Threshold: >{CROWD_THRESHOLD}"
    cv2.putText(frame, status_str,
                (px + 8, py + 91),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                (0, 60, 255) if alert_active else (130, 130, 130),
                1, cv2.LINE_AA)

    # Bottom hint bar
    ov3 = frame.copy()
    cv2.rectangle(ov3, (0, fh - 30), (fw, fh), (15, 15, 15), cv2.FILLED)
    cv2.addWeighted(ov3, 0.75, frame, 0.25, 0, frame)
    cv2.putText(frame, "Press 'S' to save screenshot  |  Press 'Q' to quit",
                (12, fh - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (180, 180, 180), 1, cv2.LINE_AA)

def main():
    print("=" * 60)
    print("  Phase 3: Crowd Alert System")
    print(f"  Alert threshold : >{CROWD_THRESHOLD} persons")
    print(f"  Alert cooldown  : {ALERT_COOLDOWN} seconds between beeps")
    print(f"  Beep frequency  : {BEEP_FREQUENCY} Hz")
    print(f"  Beep duration   : {BEEP_DURATION} ms")
    print(f"  Audio available : {AUDIO_AVAILABLE}")
    print(f"  Output folder   : {OUTPUT_DIR}")
    print("=" * 60)

    # Verify output folder before doing anything else
    if not setup_output_dir(OUTPUT_DIR):
        print("[ERROR] Cannot create or write to output folder. Exiting.")
        print("[ERROR] Try running VS Code as Administrator.")
        return

    # Load model
    print("\n[INFO] Loading YOLOv8n model...")
    model = YOLO(MODEL_PATH)
    print("[INFO] Model loaded successfully.")

    # Open camera
    print(f"\n[INFO] Opening camera (index {CAMERA_INDEX})...")
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("[ERROR] Cannot open camera. Try changing CAMERA_INDEX to 1.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    ret, test = cap.read()
    if not ret:
        print("[ERROR] Camera opened but cannot read frames.")
        cap.release()
        return

    h, w = test.shape[:2]
    print(f"[INFO] Camera ready. Resolution: {w}x{h}")
    print(f"\n[INFO] Monitoring started. Alert fires when >{CROWD_THRESHOLD} persons detected.")
    print("[INFO] Press Q to quit | S to save screenshot manually")
    print("-" * 60)

    # State variables
    fps_start       = time.time()
    fps_counter     = 0
    fps_display     = 0.0
    last_alert_time = 0.0      # time of last beep — 0 means alert fires immediately
    total_alerts    = 0
    blink_counter   = 0
    blink_on        = True
    alert_active    = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Lost camera feed.")
            break

        # Run YOLO
        results    = model(frame, conf=CONFIDENCE_MIN, verbose=False)
        detections = results[0].boxes

        # Filter persons only (class_id == 0)
        person_count  = 0
        person_number = 0

        if detections is not None:
            for box in detections:
                if int(box.cls[0]) != PERSON_CLASS_ID:
                    continue
                person_count  += 1
                person_number += 1
                draw_person_box(frame, box, person_number,
                                float(box.conf[0]), alert_active)

        # ── CROWD ALERT LOGIC ─────────────────────────────────────────────────
        now = time.time()

        if person_count > CROWD_THRESHOLD:
            alert_active = True

            # Fire alert only if cooldown has passed
            if (now - last_alert_time) >= ALERT_COOLDOWN:
                total_alerts   += 1
                last_alert_time = now

                print(f"\n[ALERT #{total_alerts}] "
                      f"{datetime.now().strftime('%H:%M:%S')} "
                      f"— {person_count} persons detected!")

                # Beep in background thread — camera does not freeze
                play_beep_async(BEEP_FREQUENCY, BEEP_DURATION)

                # Save screenshot
                saved_path = save_screenshot(frame, OUTPUT_DIR, person_count)

                # Log to CSV — safe, never crashes program even if it fails
                log_to_csv(OUTPUT_DIR, person_count, saved_path)

        else:
            alert_active = False

        # Blink effect — toggles every 15 frames
        blink_counter += 1
        if blink_counter >= 15:
            blink_counter = 0
            blink_on      = not blink_on

        # Draw everything
        if alert_active:
            draw_alert_banner(frame, person_count, blink_on)

        draw_hud(frame, person_count, fps_display, alert_active, total_alerts)

        # FPS
        fps_counter += 1
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            fps_display = fps_counter / elapsed
            fps_counter = 0
            fps_start   = time.time()

        # Show frame
        cv2.imshow("Phase 3 — Crowd Alert System | Techlive Solutions", frame)

        # Keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            print(f"\n[INFO] Session ended. Total alerts: {total_alerts}")
            print(f"[INFO] Log saved at: {OUTPUT_DIR}\\crowd_log.csv")
            break
        elif key == ord('s') or key == ord('S'):
            path = save_screenshot(frame, OUTPUT_DIR, person_count)
            print(f"[INFO] Manual screenshot: {path}")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Phase 3 complete.")
    print("=" * 60)

if __name__ == "__main__":
    main()
