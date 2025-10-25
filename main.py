import os, time, cv2, numpy as np
from ultralytics import YOLO
import mss
import csv
import win32api
import win32con
import win32gui
import win32ui
import ctypes
from ctypes import windll, c_long, c_ulong, Structure, Union, c_int, POINTER, sizeof
from threading import Thread
import tkinter as tk

# ========== CONFIG ==========
MODEL_PATH = "runs/detect/pubg_detect4/weights/best.pt"
USE_PRETRAIN = False
CONF_THRESH = 0.20  # 🔧 Giảm threshold để detect xa hơn
IMG_SZ = 480  # 🔧 Tăng lại 320 cho detection xa
MONITOR = None
OUTPUT_CSV = "targets_log.csv"
SKIP_FRAMES = 2  # 🔧 Giảm xuống 2 (vẫn 30-35 FPS nhưng detect tốt hơn)

# AIM ASSIST CONFIG
AIM_ASSIST_ENABLED = True
AIM_SMOOTH = 0.2
AIM_FOV = 250
USE_MOUSE_EVENT = True
MOUSE_DELAY = 0.001
TARGET_LOCK_DISTANCE = 250
MIN_LOCK_FRAMES = 5
USE_EMA_SMOOTHING = True
EMA_ALPHA = 0.3

# CROSSHAIR OFFSET
CROSSHAIR_OFFSET_X = 0
CROSSHAIR_OFFSET_Y = 0  # 🎯 Đặt 0 để trùng với game crosshair

# 🎯 AIM POINT CONFIG
AIM_AT_HEAD = False  # True = ngắm đầu, False = ngắm giữa body
HEAD_OFFSET_RATIO = 0.5  # 0.25 = 1/4 từ trên xuống (gần đầu)

# OVERLAY CONFIG
SHOW_BOXES = True
SHOW_FOV_CIRCLE = True  # 🎯 Bật lại FOV circle
SHOW_CROSSHAIR = True
SHOW_FPS = True
OVERLAY_OPACITY = 0.7
REDUCED_VISUALS = True  # 🚀 Giảm số lượng draw calls
GAME_WINDOW_TITLE = "PUBG: BATTLEGROUNDS"  # 🎯 Tên window game (có thể là "PUBG: BATTLEGROUNDS" hoặc "TslGame")

# ========== Mouse Setup ==========
PUL = POINTER(c_ulong)

class KeyBdInput(Structure):
    _fields_ = [("wVk", c_int), ("wScan", c_int), ("dwFlags", c_int), 
                ("time", c_int), ("dwExtraInfo", PUL)]

class HardwareInput(Structure):
    _fields_ = [("uMsg", c_int), ("wParamL", c_int), ("wParamH", c_int)]

class MouseInput(Structure):
    _fields_ = [("dx", c_long), ("dy", c_long), ("mouseData", c_int),
                ("dwFlags", c_int), ("time", c_int), ("dwExtraInfo", PUL)]

class Input_I(Union):
    _fields_ = [("ki", KeyBdInput), ("mi", MouseInput), ("hi", HardwareInput)]

class Input(Structure):
    _fields_ = [("type", c_int), ("ii", Input_I)]

# ========== TRANSPARENT OVERLAY WINDOW ==========
class TransparentOverlay:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.root = tk.Tk()
        self.root.title("PUBG Overlay")
        
        # Transparent window setup
        self.root.attributes('-topmost', True)
        self.root.attributes('-transparentcolor', 'black')
        self.root.attributes('-alpha', OVERLAY_OPACITY)
        self.root.overrideredirect(True)
        
        # Full screen
        self.root.geometry(f"{width}x{height}+0+0")
        
        # Canvas for drawing
        self.canvas = tk.Canvas(self.root, bg='black', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Make window click-through
        hwnd = windll.user32.GetParent(self.root.winfo_id())
        styles = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
        styles = win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT
        win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, styles)
        
        self.running = True
        
    def clear(self):
        """Xóa canvas"""
        self.canvas.delete("all")
    
    def draw_line(self, x1, y1, x2, y2, color="red", width=2):
        """Vẽ đường thẳng"""
        self.canvas.create_line(x1, y1, x2, y2, fill=color, width=width)
    
    def draw_rect(self, x1, y1, x2, y2, color="green", width=2):
        """Vẽ hình chữ nhật"""
        self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=width)
    
    def draw_circle(self, x, y, radius, color="yellow", width=2):
        """Vẽ hình tròn"""
        self.canvas.create_oval(x-radius, y-radius, x+radius, y+radius, 
                               outline=color, width=width)
    
    def draw_text(self, x, y, text, color="white", size=12):
        """Vẽ text"""
        self.canvas.create_text(x, y, text=text, fill=color, 
                               font=("Arial", size, "bold"), anchor="nw")
    
    def update(self):
        """Update window"""
        try:
            self.root.update()
        except:
            self.running = False
    
    def destroy(self):
        """Đóng window"""
        self.running = False
        try:
            self.root.destroy()
        except:
            pass

# ========== EMA SMOOTHER ==========
class EMASmoothing:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.smoothed_x = None
        self.smoothed_y = None
    
    def update(self, x, y):
        if self.smoothed_x is None:
            self.smoothed_x = x
            self.smoothed_y = y
        else:
            self.smoothed_x = self.alpha * x + (1 - self.alpha) * self.smoothed_x
            self.smoothed_y = self.alpha * y + (1 - self.alpha) * self.smoothed_y
        return int(self.smoothed_x), int(self.smoothed_y)
    
    def reset(self):
        self.smoothed_x = None
        self.smoothed_y = None

# ========== Helper Functions ==========
def move_mouse_relative(dx, dy):
    try:
        windll.user32.mouse_event(0x0001, int(dx), int(dy), 0, 0)
        return True
    except:
        return False

def move_mouse_sendinput(dx, dy):
    try:
        extra = c_ulong(0)
        ii_ = Input_I()
        ii_.mi = MouseInput(int(dx), int(dy), 0, 0x0001, 0, POINTER(c_ulong)(extra))
        x = Input(0, ii_)
        windll.user32.SendInput(1, POINTER(Input)(x), sizeof(x))
        return True
    except:
        return False

def smooth_move_mouse(current_x, current_y, target_x, target_y, smooth_factor, dist_to_target):
    dx = target_x - current_x
    dy = target_y - current_y
    
    if dist_to_target < 10:
        adaptive_smooth = smooth_factor * 0.5
    elif dist_to_target < 30:
        adaptive_smooth = smooth_factor * 0.7
    elif dist_to_target < 100:
        adaptive_smooth = smooth_factor * 1.0
    else:
        adaptive_smooth = smooth_factor * 1.5
    
    move_x = int(dx * adaptive_smooth)
    move_y = int(dy * adaptive_smooth)
    
    if abs(move_x) >= 1 or abs(move_y) >= 1:
        if USE_MOUSE_EVENT:
            success = move_mouse_relative(move_x, move_y)
            if not success:
                success = move_mouse_sendinput(move_x, move_y)
        else:
            new_x = current_x + move_x
            new_y = current_y + move_y
            ctypes.windll.user32.SetCursorPos(new_x, new_y)
            success = True
        
        time.sleep(MOUSE_DELAY)
        return success
    return False

def get_best_target(dets, screen_center_x, screen_center_y, fov_radius, locked_target=None):
    if not dets:
        return None
    
    valid_targets = []
    for d in dets:
        cx, cy = d['center']
        dist_from_center = np.sqrt((cx - screen_center_x)**2 + (cy - screen_center_y)**2)
        
        if dist_from_center <= fov_radius:
            score = (d['conf'] * 2.0) / (dist_from_center + 1)
            valid_targets.append((d, score, dist_from_center))
    
    if not valid_targets:
        return None
    
    if locked_target is not None:
        locked_cx, locked_cy = locked_target['center']
        for d, score, dist_from_center in valid_targets:
            cx, cy = d['center']
            target_dist = np.sqrt((cx - locked_cx)**2 + (cy - locked_cy)**2)
            
            if target_dist < TARGET_LOCK_DISTANCE:
                return d
    
    best = max(valid_targets, key=lambda x: x[1])
    return best[0]

def check_shift_pressed():
    left_shift = win32api.GetAsyncKeyState(0xA0) & 0x8000
    right_shift = win32api.GetAsyncKeyState(0xA1) & 0x8000
    return left_shift != 0 or right_shift != 0

def check_key_pressed(vk_code):
    return win32api.GetAsyncKeyState(vk_code) & 0x8000 != 0

def is_game_window_active():
    """Kiểm tra xem game có đang active không"""
    try:
        # Lấy window đang active
        hwnd = win32gui.GetForegroundWindow()
        window_title = win32gui.GetWindowText(hwnd)
        
        # Check nếu title chứa tên game
        game_keywords = ["PUBG", "TslGame", "BATTLEGROUNDS"]
        for keyword in game_keywords:
            if keyword.lower() in window_title.lower():
                return True
        return False
    except:
        return False

# ========== Setup Model ==========
os.chdir(os.path.dirname(os.path.abspath(__file__)))

if USE_PRETRAIN:
    model = YOLO("yolov8n.pt")
    print("Using pretrained YOLOv8n model")
else:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print(f"Loaded custom model: {MODEL_PATH}")

model.overrides['verbose'] = False
model.overrides['half'] = True  # 🚀 Bật FP16 inference (nhanh hơn ~30%)
model.overrides['device'] = 0  # 🚀 Force GPU nếu có

print(f"Model classes: {model.names}")
print(f"\n🎮 OVERLAY MODE - Drawing directly on screen!")
print(f"Game Detection: Will only show overlay when PUBG is active")
print(f"Aim Assist: {'ENABLED' if AIM_ASSIST_ENABLED else 'DISABLED'}")
print(f"Aim Target: {'HEAD (top 25%)' if AIM_AT_HEAD else 'BODY CENTER'}")
print(f"EMA Smoothing: {'ENABLED' if USE_EMA_SMOOTHING else 'DISABLED'}")
print("\n=== CONTROLS ===")
print("SHIFT (hold)  : Activate aim assist")
print("F1            : Toggle aim assist ON/OFF")
print("F2            : Toggle boxes")
print("F3            : Toggle FOV circle")
print("F4            : Toggle crosshair")
print("F5            : Toggle FPS display")
print("F6            : Toggle EMA smoothing")
print("F7            : Toggle HEAD/BODY aim")
print("F8            : Toggle AIM ON/OFF (same as F1)")
print("ESC           : Quit")
print("================\n")

csv_file = open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["timestamp", "shift", "target_x", "target_y", "conf", "class", "aimed", "method"])

# ========== MAIN LOOP ==========
with mss.mss() as sct:
    monitor = sct.monitors[1] if MONITOR is None else MONITOR
    print(f"Capturing screen: {monitor}")
    
    screen_w = monitor['width']
    screen_h = monitor['height']
    screen_center_x = screen_w // 2 + CROSSHAIR_OFFSET_X
    screen_center_y = screen_h // 2 + CROSSHAIR_OFFSET_Y
    
    monitor_left = monitor['left']
    monitor_top = monitor['top']
    
    print(f"Screen size: {screen_w}x{screen_h}")
    print(f"Creating overlay window...\n")
    
    # Create overlay
    overlay = TransparentOverlay(screen_w, screen_h)
    
    last_time = time.time()
    fps_smooth = 0.0
    aim_debug_log = []
    frame_count = 0
    locked_target = None
    lock_frames = 0
    ema_smoother = EMASmoothing(alpha=EMA_ALPHA)
    
    # 🚀 Cache detections
    cached_dets = []
    last_inference_time = 0
    
    # Create overlay
    overlay = TransparentOverlay(screen_w, screen_h)
    
    try:
        while overlay.running:
            frame_count += 1
            t_frame_start = time.time()
            
            # 🎯 KIỂM TRA GAME CÓ ACTIVE KHÔNG
            game_is_active = is_game_window_active()
            
            # Check hotkeys (giảm frequency)
            if check_key_pressed(0x1B):  # ESC
                print("\nExiting...")
                break
            if check_key_pressed(0x70):  # F1
                AIM_ASSIST_ENABLED = not AIM_ASSIST_ENABLED
                print(f"AIM: {AIM_ASSIST_ENABLED}")
                time.sleep(0.2)
            if check_key_pressed(0x71):  # F2
                SHOW_BOXES = not SHOW_BOXES
                print(f"BOXES: {SHOW_BOXES}")
                time.sleep(0.2)
            if check_key_pressed(0x72):  # F3
                SHOW_FOV_CIRCLE = not SHOW_FOV_CIRCLE
                print(f"FOV: {SHOW_FOV_CIRCLE}")
                time.sleep(0.2)
            if check_key_pressed(0x73):  # F4
                SHOW_CROSSHAIR = not SHOW_CROSSHAIR
                print(f"CROSSHAIR: {SHOW_CROSSHAIR}")
                time.sleep(0.2)
            if check_key_pressed(0x74):  # F5
                SHOW_FPS = not SHOW_FPS
                print(f"FPS: {SHOW_FPS}")
                time.sleep(0.2)
            if check_key_pressed(0x75):  # F6
                USE_EMA_SMOOTHING = not USE_EMA_SMOOTHING
                print(f"EMA: {USE_EMA_SMOOTHING}")
                time.sleep(0.2)
            if check_key_pressed(0x76):  # F7
                AIM_AT_HEAD = not AIM_AT_HEAD
                target_text = "HEAD" if AIM_AT_HEAD else "BODY"
                print(f"AIM TARGET: {target_text}")
                time.sleep(0.2)
            if check_key_pressed(0x77):  # F8
                AIM_ASSIST_ENABLED = not AIM_ASSIST_ENABLED
                print(f"AIM ASSIST: {AIM_ASSIST_ENABLED}")
                time.sleep(0.2)
            
            # Capture screen
            sct_img = sct.grab(monitor)
            frame = np.array(sct_img)
            if frame.shape[2] == 4:
                frame = frame[:, :, :3]
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            h0, w0 = frame.shape[:2]
            
            # Scale for inference
            scale = IMG_SZ / max(w0, h0)
            if scale != 1.0:
                small = cv2.resize(frame, (int(w0*scale), int(h0*scale)))
            else:
                small = frame
            
            # YOLO Inference
            results = model.predict(small, conf=CONF_THRESH, imgsz=IMG_SZ, verbose=False)
            
            dets = []
            if len(results) > 0:
                r = results[0]
                boxes = r.boxes
                for box in boxes:
                    conf = float(box.conf.cpu().numpy())
                    cls = int(box.cls.cpu().numpy())
                    xyxy = box.xyxy.cpu().numpy().astype(int).tolist()[0]
                    x1, y1, x2, y2 = xyxy
                    
                    inv = 1 / scale
                    x1 = int(x1 * inv)
                    y1 = int(y1 * inv)
                    x2 = int(x2 * inv)
                    y2 = int(y2 * inv)
                    
                    # 🎯 Tính aim point
                    cx = int((x1 + x2) / 2)  # Giữa theo X
                    
                    if AIM_AT_HEAD:
                        # Ngắm đầu: offset từ trên xuống
                        box_height = y2 - y1
                        cy = int(y1 + box_height * HEAD_OFFSET_RATIO)
                    else:
                        # Ngắm giữa body
                        cy = int((y1 + y2) / 2)
                    
                    dets.append({
                        'cls': cls,
                        'conf': conf,
                        'xyxy': (x1, y1, x2, y2),
                        'center': (cx, cy)
                    })
            
            # Clear overlay
            overlay.clear()
            
            # Draw FOV circle
            if SHOW_FOV_CIRCLE:
                overlay.draw_circle(screen_center_x, screen_center_y, AIM_FOV, "gray", 2)
            
            # Draw crosshair
            if SHOW_CROSSHAIR:
                overlay.draw_line(screen_center_x-15, screen_center_y, 
                                screen_center_x+15, screen_center_y, "white", 2)
                overlay.draw_line(screen_center_x, screen_center_y-15, 
                                screen_center_x, screen_center_y+15, "white", 2)
                overlay.draw_circle(screen_center_x, screen_center_y, 3, "white", 2)
            
            # Draw detections
            if SHOW_BOXES:
                names = model.names
                for d in dets:
                    x1, y1, x2, y2 = d['xyxy']
                    cx, cy = d['center']
                    conf = d['conf']
                    cls = d['cls']
                    
                    dist_from_crosshair = np.sqrt((cx - screen_center_x)**2 + 
                                                 (cy - screen_center_y)**2)
                    in_fov = dist_from_crosshair <= AIM_FOV
                    
                    color = "lime" if in_fov else "gray"
                    
                    # 🚀 Giảm draw calls nếu REDUCED_VISUALS
                    if REDUCED_VISUALS:
                        if in_fov:  # Chỉ vẽ target trong FOV
                            overlay.draw_rect(x1, y1, x2, y2, color, 2)
                            overlay.draw_circle(cx, cy, 5, "red", 2)
                    else:
                        overlay.draw_rect(x1, y1, x2, y2, color, 2)
                        overlay.draw_circle(cx, cy, 5, "red", 2)
                        label = f"{names[cls]}:{conf:.2f}"
                        overlay.draw_text(x1, y1-20, label, color, 10)
            
            # AIM ASSIST
            shift_held = check_shift_pressed()
            aimed = False
            method_used = "none"
            
            if shift_held and AIM_ASSIST_ENABLED and dets:
                target = get_best_target(dets, screen_center_x, screen_center_y, 
                                        AIM_FOV, locked_target)
                
                if target:
                    tx_raw, ty_raw = target['center']
                    
                    if USE_EMA_SMOOTHING:
                        tx, ty = ema_smoother.update(tx_raw, ty_raw)
                    else:
                        tx, ty = tx_raw, ty_raw
                    
                    if locked_target is not None:
                        old_tx, old_ty = locked_target['center']
                        target_change_dist = np.sqrt((tx_raw - old_tx)**2 + 
                                                    (ty_raw - old_ty)**2)
                        
                        if target_change_dist < TARGET_LOCK_DISTANCE:
                            lock_frames += 1
                        else:
                            lock_frames = 0
                            ema_smoother.reset()
                    else:
                        lock_frames = 0
                        ema_smoother.reset()
                    
                    locked_target = target.copy()
                    locked_target['center'] = (tx_raw, ty_raw)
                    
                    # Draw lock indicator
                    lock_color = "lime" if lock_frames >= MIN_LOCK_FRAMES else "orange"
                    
                    # 🚀 Simplified visuals
                    if REDUCED_VISUALS:
                        # Chỉ vẽ cross và circle
                        overlay.draw_line(tx-20, ty, tx+20, ty, lock_color, 3)
                        overlay.draw_line(tx, ty-20, tx, ty+20, lock_color, 3)
                    else:
                        # Full visuals
                        overlay.draw_circle(tx_raw, ty_raw, 5, "magenta", 2)
                        overlay.draw_line(tx-25, ty, tx+25, ty, lock_color, 3)
                        overlay.draw_line(tx, ty-25, tx, ty+25, lock_color, 3)
                        overlay.draw_circle(tx, ty, 10, lock_color, 2)
                        
                        lock_text = f"LOCKED #{lock_frames}" if lock_frames >= MIN_LOCK_FRAMES else "TRACK"
                        overlay.draw_text(tx+15, ty-15, lock_text, lock_color, 12)
                    
                    # Aim line
                    overlay.draw_line(screen_center_x, screen_center_y, tx, ty, "cyan", 2)
                    
                    # Mouse movement
                    target_screen_x = monitor_left + tx
                    target_screen_y = monitor_top + ty
                    
                    cursor_pos = win32api.GetCursorPos()
                    cursor_x, cursor_y = cursor_pos
                    
                    dist_to_target = np.sqrt((target_screen_x - cursor_x)**2 + 
                                            (target_screen_y - cursor_y)**2)
                    
                    if lock_frames >= MIN_LOCK_FRAMES and dist_to_target > 3:
                        aimed = smooth_move_mouse(cursor_x, cursor_y,
                                                target_screen_x, target_screen_y,
                                                AIM_SMOOTH, dist_to_target)
                        method_used = "mouse_event" if USE_MOUSE_EVENT else "SetCursorPos"
                    
                    csv_writer.writerow([time.time(), True, tx, ty,
                                        target['conf'], names[target['cls']], 
                                        aimed, method_used])
                else:
                    locked_target = None
                    lock_frames = 0
                    ema_smoother.reset()
            else:
                locked_target = None
                lock_frames = 0
                ema_smoother.reset()
                csv_writer.writerow([time.time(), shift_held, '', '', '', '', False, 'none'])
            
            # FPS
            t1 = time.time()
            fps = 1.0 / (t1 - last_time) if (t1 - last_time) > 0 else 0.0
            last_time = t1
            fps_smooth = fps_smooth * 0.9 + fps * 0.1  # 🚀 Smooth hơn
            
            # Draw status - 🎯 GIỮA MÀN HÌNH THEO HÀNG NGANG
            if SHOW_FPS:
                # Vị trí giữa màn hình, phía trên
                center_x = screen_w // 2
                status_y = 30
                
                # Tạo status string
                fps_text = f"FPS: {int(fps_smooth)}"
                aim_text = f"AIM: {'ON' if AIM_ASSIST_ENABLED else 'OFF'}"
                shift_text = f"SHIFT: {'HELD' if shift_held else 'OFF'}"
                target_text = f"Target: {'HEAD' if AIM_AT_HEAD else 'BODY'}"
                ema_text = f"EMA: {'ON' if USE_EMA_SMOOTHING else 'OFF'}"
                
                # Ghép thành 1 dòng
                full_status = f"{fps_text}  |  {aim_text}  |  {shift_text}  |  {target_text}  |  {ema_text}"
                
                # Tính độ rộng để center
                text_width = len(full_status) * 8
                start_x = center_x - (text_width // 2)
                
                # Vẽ background đen mờ cho dễ đọc
                bg_padding = 10
                overlay.canvas.create_rectangle(
                    start_x - bg_padding, status_y - bg_padding,
                    start_x + text_width + bg_padding, status_y + 25,
                    fill='black', stipple='gray50'
                )
                
                # Vẽ text với màu tương ứng
                x_pos = start_x
                
                # FPS
                overlay.draw_text(x_pos, status_y, fps_text, "yellow", 14)
                x_pos += len(fps_text) * 8 + 16
                overlay.draw_text(x_pos, status_y, "|", "gray", 14)
                x_pos += 24
                
                # AIM
                aim_color = "lime" if AIM_ASSIST_ENABLED else "red"
                overlay.draw_text(x_pos, status_y, aim_text, aim_color, 14)
                x_pos += len(aim_text) * 8 + 16
                overlay.draw_text(x_pos, status_y, "|", "gray", 14)
                x_pos += 24
                
                # SHIFT
                shift_color = "cyan" if shift_held else "gray"
                overlay.draw_text(x_pos, status_y, shift_text, shift_color, 14)
                x_pos += len(shift_text) * 8 + 16
                overlay.draw_text(x_pos, status_y, "|", "gray", 14)
                x_pos += 24
                
                # TARGET
                aim_target_color = "orange" if AIM_AT_HEAD else "cyan"
                overlay.draw_text(x_pos, status_y, target_text, aim_target_color, 14)
                x_pos += len(target_text) * 8 + 16
                overlay.draw_text(x_pos, status_y, "|", "gray", 14)
                x_pos += 24
                
                # EMA
                ema_color = "lime" if USE_EMA_SMOOTHING else "gray"
                overlay.draw_text(x_pos, status_y, ema_text, ema_color, 14)
            
            # Update overlay
            overlay.update()

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Stopping...")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        overlay.destroy()
        csv_file.close()
        print(f"\nCSV log saved: {OUTPUT_CSV}")
        print("Session complete!")