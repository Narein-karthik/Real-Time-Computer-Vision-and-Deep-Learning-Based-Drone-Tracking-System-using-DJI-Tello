from djitellopy import tello
import cv2 as opencv
import keyboard
from time import sleep, time
import datetime
import os
from ultralytics import YOLO
import torch

# --- Tello and YOLO Setup ---
me = tello.Tello()
me.connect()
print(f"Battery life: {me.get_battery()}%")

me.streamon()

# Load the YOLOv8 model.
model = YOLO('C:/Users/Narein karthik.E/Downloads/best (3).pt')

# --- GPU SETUP: Identify and set device ---
if torch.cuda.is_available():
    device = 'cuda'
    print("Model will run on GPU (CUDA).")
else:
    device = 'cpu'
    print("Model running on CPU (No CUDA device found).")

# Move the model weights to the selected device
model.to(device)


# --- TARGET CONFIGURATION ---
TARGET_CLASSES = ["Face"]

# --- DIAGNOSTIC: PRINT MODEL CLASS NAMES ---
try:
    print(f"\n--- CUSTOM MODEL CLASS NAMES ---")
    print(f"Your model contains {len(model.names)} classes: {model.names}")
    print(f"----------------------------------\n")
except Exception as e:
    print(f"Warning: Could not read model names. Error: {e}")

# --- Video Recording Setup ---
fourcc = opencv.VideoWriter_fourcc(*'mp4v')
frame_width = 640
frame_height = 480
output_filename = f"tello_recording_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
output_folder = "tello_recordings"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

output_path = os.path.join(output_folder, output_filename)
# Set VideoWriter FPS to 15 to match the stable capture rate
out = opencv.VideoWriter(output_path, fourcc, 15, (frame_width, frame_height))

print(f"Recording video to: {output_path}")

# --- Drone Control Variables ---
is_tracking = False
fb_speed = 0
lr_speed = 0
ud_speed = 0
yv_speed = 0

# Proportional gain constants for the PID-like controller (SOFTENED FOR STABILITY)
PID_P_YAW = 0.20
PID_P_UD = 0.20
PID_P_FB = 0.08    # CRITICAL: Softest gain for stable distance control

MAX_FB_SPEED = 35

# Target Area reduced to force the drone to move farther away
TARGET_AREA = 35000

# ***VERTICAL OFFSET ADDED HERE (This is the negative value to force the drone to look down)***
VERTICAL_OFFSET = -40

MIN_AREA_THRESHOLD = 5000
MAX_AREA_THRESHOLD = 150000

# DEAD ZONE INCREASED for tolerance
FB_DEADBAND = 15000


# Function to get keyboard input and handle modes
def get_keyboard_input():
    global is_tracking, fb_speed, lr_speed, ud_speed, yv_speed

    # Reset speeds
    lr_speed, fb_speed, ud_speed, yv_speed = 0, 0, 0, 0
    manual_override_speed = 50

    # Toggle tracking mode
    if keyboard.is_pressed("t"):
        is_tracking = not is_tracking
        if is_tracking:
            print("Tracking mode engaged.")
        else:
            print("Tracking mode disengaged.")
        sleep(0.5)

    # Check for manual override. If any manual key is pressed, disable tracking.
    if keyboard.is_pressed("left") or keyboard.is_pressed("right") or \
            keyboard.is_pressed("up") or keyboard.is_pressed("down") or \
            keyboard.is_pressed("w") or keyboard.is_pressed("s") or \
            keyboard.is_pressed("a") or keyboard.is_pressed("d"):
        is_tracking = False
        print("Manual override. Tracking disengaged.")

    if not is_tracking:
        # Manual control logic
        if keyboard.is_pressed("left"):
            lr_speed = -manual_override_speed
        elif keyboard.is_pressed("right"):
            lr_speed = manual_override_speed

        if keyboard.is_pressed("up"):
            fb_speed = manual_override_speed
        elif keyboard.is_pressed("down"):
            fb_speed = -manual_override_speed

        if keyboard.is_pressed("w"):
            ud_speed = manual_override_speed
        elif keyboard.is_pressed("s"):
            ud_speed = -manual_override_speed

        if keyboard.is_pressed("a"):
            yv_speed = -manual_override_speed
        elif keyboard.is_pressed("d"):
            yv_speed = manual_override_speed

    # Separate takeoff and land commands
    if keyboard.is_pressed("q"):
        me.land()
        sleep(3)
    if keyboard.is_pressed("e"):
        me.takeoff()
        sleep(3)


def track_object(img):
    global fb_speed, lr_speed, ud_speed, yv_speed

    # Reset speeds before calculating new ones
    fb_speed, lr_speed, ud_speed, yv_speed = 0, 0, 0, 0

    # Use the track method to get results
    # Set imgsz=320 for speed, pass 'device' for GPU, and set 'conf' for filtering
    results = model.track(img, persist=True, verbose=False, conf=0.5, device=device, imgsz=320)

    # --- Target Selection Logic (Highest Confidence) ---
    target = None
    max_confidence = -1.0

    if results and results[0].boxes and results[0].boxes.data.numel() > 0:

        for box in results[0].boxes.data:
            cls_id = int(box[-1].item())
            confidence = box[-2].item()

            try:
                detected_name = model.names[cls_id]
            except:
                detected_name = f"Class {cls_id}"

            # 1. Check if the object is a target class (must be in TARGET_CLASSES list)
            if detected_name in TARGET_CLASSES:
                if confidence > max_confidence:
                    max_confidence = confidence
                    target = box

    if target is not None:
        x1, y1, x2, y2 = map(int, target[:4])

        # Calculate bounding box area and center
        box_area = (x2 - x1) * (y2 - y1)
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # Draw bounding box and information
        opencv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        opencv.putText(img, f'Tracking: {model.names[int(target[-1].item())]} ({max_confidence:.2f})',
                    (x1, y1 - 10), opencv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        opencv.circle(img, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)

        # Calculate error from frame center for rotation (yaw) and up/down movement
        error_yaw = center_x - frame_width / 2

        # --- VERTICAL OFFSET CORRECTION LOGIC (Uses the defined -40 offset) ---
        target_y = frame_height / 2 + VERTICAL_OFFSET
        error_ud = center_y - target_y

        # Check if the detected object's area is within the safe tracking range
        if MIN_AREA_THRESHOLD < box_area < MAX_AREA_THRESHOLD:
            # Proportional control for yaw (left/right rotation)
            yv_speed = int(error_yaw * PID_P_YAW)

            # Proportional control for up/down movement
            ud_speed = int(-error_ud * PID_P_UD)

            # Proportional control for forward/backward movement (to maintain distance)
            error_fb = TARGET_AREA - box_area

            # Use a deadband to prevent small, jittery movements
            if abs(error_fb) > FB_DEADBAND:
                # Uses the softened PID_P_FB=0.08
                fb_speed = int(error_fb * PID_P_FB)

            # Clamp all speeds to Tello's valid range (-100 to 100)
            yv_speed = max(-100, min(100, yv_speed))
            ud_speed = max(-100, min(100, ud_speed))
            # Clamp the forward/backward speed to the new, safer maximum
            fb_speed = max(-MAX_FB_SPEED, min(MAX_FB_SPEED, fb_speed))

        # Adjust speeds for Tello control
        me.send_rc_control(lr_speed, fb_speed, ud_speed, yv_speed)

    else:
        # If no target is detected, hover
        me.send_rc_control(0, 0, 0, 0)


# --- Main Loop ---
# Removed pTime initialization
# Set target time to 1/15 (0.0667s) to match achievable throughput
target_frame_time = 1 / 15

while True:
    startTime = time() # Used for the Dynamic Sleep Timer

    get_keyboard_input()

    img = me.get_frame_read().frame
    # Check if a frame was received successfully
    if img is not None:
        img = opencv.cvtColor(img, opencv.COLOR_RGB2BGR)
        img = opencv.resize(img, (frame_width, frame_height))

        # --- NOISE REDUCTION FILTER ---
        # NOTE: Filter is removed/commented out to preserve FPS
        # img = opencv.bilateralFilter(img, 9, 75, 75)

        # FPS Calculation variables (cTime, pTime) and display are removed here

        if is_tracking:
            track_object(img)
            opencv.putText(img, "Tracking Active", (10, 30), opencv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            me.send_rc_control(lr_speed, fb_speed, ud_speed, yv_speed)
            opencv.putText(img, "Manual Control", (10, 30), opencv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        out.write(img)
        opencv.imshow("Tello Video - Recording...", img)
    else:
        # Re-initialize stream if the drone's video feed is lost
        me.streamoff()
        me.streamon()
        print("Tello stream disconnected. Reconnecting...")

    # Check for exit key
    if opencv.waitKey(1) & 0xFF == 27 or keyboard.is_pressed("esc"):
        me.land()
        break

    # --- Dynamic Sleep for Stability ---
    # Calculate time spent processing this frame
    elapsedTime = time() - startTime
    # Calculate remaining sleep time to maintain target FPS (1/15 = 0.0667 seconds)
    time_to_sleep = max(0, target_frame_time - elapsedTime)
    sleep(time_to_sleep)

# --- Cleanup ---
out.release()
opencv.destroyAllWindows()
me.streamoff()
print(f"Video recording saved to: {output_path}")
