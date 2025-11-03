import cv2
import json
import time
import math
import argparse
import numpy as np
from collections import deque
from pathlib import Path
import mediapipe as mp

ARM_ABILITY_VERSION = "1.3.2-bounds"

# =========================
# Paths / Config
# =========================
OUT_DIR = Path("outputs"); OUT_DIR.mkdir(parents=True, exist_ok=True)
CALIB_PATH = Path("arm_calibration.json")
CSV_PATH = OUT_DIR / "arm_ability.csv"

SMOOTH_N_ANGLE = 5
SMOOTH_N_PCT   = 5
VIS_THRESH = 0.50

FINGERS = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
JOINTS  = ["Wrist", "Elbow", "Shoulder"]
ALL_KEYS = JOINTS + FINGERS

POSE_IDX = {
    "Left":  {"shoulder":11, "elbow":13, "wrist":15, "hip":23},
    "Right": {"shoulder":12, "elbow":14, "wrist":16, "hip":24},
}

# ============== Utils ==============
def clamp_percent(p): return max(0.0, min(100.0, p))

def angle_between(v1, v2):
    v1 = np.asarray(v1, np.float32); v2 = np.asarray(v2, np.float32)
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0: return np.nan
    c = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return math.degrees(math.acos(c))

def movavg(buf: deque, val: float, maxlen: int):
    if not np.isnan(val):
        buf.append(val)
        while len(buf) > maxlen: buf.popleft()
    arr = [v for v in buf if not np.isnan(v)]
    return float(np.mean(arr)) if arr else np.nan

def robust_add(buf: deque, val: float, maxlen: int):
    # simple outlier guard (median ±45°)
    if len(buf) >= 3:
        med = float(np.median(buf))
        if not np.isnan(val) and abs(val - med) > 45.0:
            return movavg(buf, np.nan, maxlen)
    return movavg(buf, val, maxlen)

def save_calib(data: dict): CALIB_PATH.write_text(json.dumps(data, indent=2))
def load_calib():
    if CALIB_PATH.exists(): return json.loads(CALIB_PATH.read_text())
    return None

def pct_from(ang, n, m):
    """
    Direction-agnostic mapping using bounds:
      low  = min(n, m) -> 0%
      high = max(n, m) -> 100%
    Values below low clamp to 0, above high clamp to 100 in the UI.
    Returns NaN if invalid inputs.
    """
    if ang is None or n is None or m is None: return np.nan
    if np.isnan(ang) or np.isnan(n) or np.isnan(m): return np.nan
    lo = min(n, m)
    hi = max(n, m)
    denom = (hi - lo)
    if denom < 1e-3:  # avoid divide-by-near-zero
        return np.nan
    # raw percentage (unclamped) so we can still show out-of-range markers
    return 100.0 * ((ang - lo) / denom)

# ============== Landmark helpers ==============
def pt_xy(lm, i, W, H):
    if lm is None: return np.array([np.nan, np.nan], np.float32)
    if hasattr(lm, "landmark"): lm = lm.landmark
    p = lm[i]
    return np.array([p.x*W, p.y*H], dtype=np.float32)

def pt_world(lm, i):
    if lm is None: return np.array([np.nan, np.nan, np.nan], np.float32), 0.0
    p = lm[i]
    return np.array([p.x, p.y, p.z], dtype=np.float32), getattr(p, "visibility", 1.0)

def finger_ip_pip_angle(hand_lm, W, H, which: str):
    p = lambda idx: pt_xy(hand_lm, idx, W, H)
    if which == "Thumb":
        mcp, ip_, tip = p(2), p(3), p(4)
        return angle_between(mcp - ip_, tip - ip_)
    m = {"Index":(5,6,7), "Middle":(9,10,11), "Ring":(13,14,15), "Pinky":(17,18,19)}
    if which in m:
        mcp, pip_, dip = p(m[which][0]), p(m[which][1]), p(m[which][2])
        return angle_between(mcp - pip_, dip - pip_)
    return np.nan

def wrist_flexion_angle(pose_world, hand_lm, side: str, W, H):
    if pose_world is None or hand_lm is None or side is None:
        return np.nan
    e, ve = pt_world(pose_world, POSE_IDX[side]["elbow"])
    w, vw = pt_world(pose_world, POSE_IDX[side]["wrist"])
    if min(ve, vw) < VIS_THRESH: return np.nan
    forearm = w - e
    p0 = pt_xy(hand_lm, 0, W, H)
    mid_mcp = pt_xy(hand_lm, 9, W, H)
    hand_dir = np.array([mid_mcp[0]-p0[0], mid_mcp[1]-p0[1], 0.0], dtype=np.float32)
    return angle_between(forearm, hand_dir)

def elbow_flexion_angle_world(pose_world, side: str):
    if pose_world is None or side is None: return np.nan
    s, vs = pt_world(pose_world, POSE_IDX[side]["shoulder"])
    e, ve = pt_world(pose_world, POSE_IDX[side]["elbow"])
    w, vw = pt_world(pose_world, POSE_IDX[side]["wrist"])
    if min(vs, ve, vw) < VIS_THRESH: return np.nan
    ua = s - e; fa = w - e
    return angle_between(ua, fa)

def shoulder_abduction_angle_world(pose_world, side: str):
    if pose_world is None or side is None: return np.nan
    sh, vsh = pt_world(pose_world, POSE_IDX[side]["shoulder"])
    hip, vhp = pt_world(pose_world, POSE_IDX[side]["hip"])
    elb, vel = pt_world(pose_world, POSE_IDX[side]["elbow"])
    if min(vsh, vhp, vel) < VIS_THRESH: return np.nan
    torso = sh - hip; upper = elb - sh
    torso_xy = torso.copy(); torso_xy[2] = 0.0
    upper_xy = upper.copy(); upper_xy[2] = 0.0
    return angle_between(torso_xy, upper_xy)

# ============== Main ==============
def main():
    print(f"[BOOT] Arm Ability version: {ARM_ABILITY_VERSION}")

    ap = argparse.ArgumentParser()
    ap.add_argument("--side", choices=["left","right","auto"], default="auto")
    ap.add_argument("--selfie", choices=["on","off"], default="on",
                    help="Mirror the preview like a selfie")
    ap.add_argument("--labelby", choices=["subject","screen"], default="subject",
                    help="Show Left/Right by subject body or screen view")
    args = ap.parse_args()
    side_pref = {"left":"Left","right":"Right","auto":"auto"}[args.side]
    selfie = (args.selfie == "on")
    label_by_screen = (args.labelby == "screen")

    mp_hol = mp.solutions.holistic
    mp_draw = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    hol = mp_hol.Holistic(
        static_image_mode=False,
        model_complexity=1,
        refine_face_landmarks=False,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam"); return
    cam_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if cam_fps <= 1: cam_fps = 30.0

    # Calibration stores
    neutral = {k:None for k in ALL_KEYS}
    maxpos  = {k:None for k in ALL_KEYS}
    prev = load_calib()
    if prev:
        for k in neutral: neutral[k] = prev.get("neutral", {}).get(k, None)
        for k in maxpos : maxpos[k]  = prev.get("maxpos",  {}).get(k, None)
        print("[INFO] Loaded arm calibration from arm_calibration.json")

    angle_s = {k: deque(maxlen=SMOOTH_N_ANGLE) for k in ALL_KEYS}
    pct_s   = {k: deque(maxlen=SMOOTH_N_PCT)   for k in ALL_KEYS}

    # Session
    recording_csv = False; csv_file = None
    recording_video = False; writer = None; out_path = None
    paused = False
    locked_side = None
    start_time = time.time()
    session_reps = 0
    show_diag = False  # diagnostics panel toggle

    # Targets
    target_idx = 0
    target_pct = {k: None for k in ALL_KEYS}

    # Rep counting
    rep_metrics_cycle = JOINTS + FINGERS
    rep_metric_i = 1  # default Elbow
    rep_state = 0
    up_thr = 60.0
    down_thr = 20.0

    # Per-finger calibration selection
    sel_finger_i = 0
    def current_finger_name():
        return FINGERS[sel_finger_i % len(FINGERS)]

    # Wizard (10s per step)
    wizard_on = False
    wizard_steps = [
        ("HAND Neutral",   ["Wrist"] + FINGERS, "Open your hand fully.\nKeep wrist relaxed and straight."),
        ("HAND Max",       ["Wrist"] + FINGERS, "Make a tight fist.\nFlex wrist forward comfortably."),
        ("ARM Neutral",    ["Elbow","Shoulder"], "Let your arm rest by your side.\nStand tall, shoulders relaxed."),
        ("ARM Max",        ["Elbow","Shoulder"], "Bend your elbow as far as comfortable.\nRaise arm out to the side."),
    ]
    wizard_i = 0
    wizard_countdown_end = 0.0
    WARMUP_SECONDS = 10.0

    print(f"""
Controls:
  W - start/stop 10s guided warm-up (HAND N → HAND M → ARM N → ARM M)
  1/2 - HAND Neutral/Max (burst)    | 3/4 - ARM Neutral/Max (burst)
  [ / ] - select finger              | 5 / 6 - selected FINGER Neutral/Max (burst)
  N/M - ALL Neutral/Max (burst over all)
  T - cycle target metric, Up/Down arrows to set target % (±5%)
  Y - cycle repetition metric (default Elbow)
  D - toggle diagnostics
  P - pause/resume | K - snapshot PNG
  R - CSV log toggle | V - Video record toggle | S - Save calibration | Q - Quit

Side: {args.side.upper()}  |  Selfie: {args.selfie.upper()}  |  Labels: {args.labelby.upper()}
""")

    # ---- Burst average helper (raw angles, not %)
    def burst_average(keys, bursts=12):
        """Average a short burst of raw angles for the given keys."""
        acc = {k: np.nan for k in keys}
        for _ in range(bursts):
            ok2, fr2 = cap.read()
            if not ok2: break
            if selfie: fr2 = cv2.flip(fr2, 1)
            r2 = hol.process(cv2.cvtColor(fr2, cv2.COLOR_BGR2RGB))
            # decide subject-side
            s2 = None
            if side_pref == "auto":
                if r2.right_hand_landmarks is not None: s2 = "Right"
                elif r2.left_hand_landmarks is not None: s2 = "Left"
            else:
                s2 = "Right" if side_pref == "Right" else "Left"
            pose_w2 = r2.pose_world_landmarks.landmark if r2.pose_world_landmarks is not None else None
            hand2 = r2.right_hand_landmarks if s2=="Right" else (r2.left_hand_landmarks if s2=="Left" else None)

            vals = {}
            vals["Wrist"]    = wrist_flexion_angle(pose_w2, hand2, s2, fr2.shape[1], fr2.shape[0]) if s2 else np.nan
            vals["Elbow"]    = elbow_flexion_angle_world(pose_w2, s2) if s2 else np.nan
            vals["Shoulder"] = shoulder_abduction_angle_world(pose_w2, s2) if s2 else np.nan
            if hand2 is not None:
                for f in FINGERS:
                    vals[f] = finger_ip_pip_angle(hand2, fr2.shape[1], fr2.shape[0], f)
            for k in keys:
                acc[k] = np.nanmean([acc[k], vals.get(k, np.nan)])
        return {k: (float(acc[k]) if not np.isnan(acc[k]) else None) for k in keys}

    while True:
        ok, frame = cap.read()
        if not ok: break

        if selfie:
            frame = cv2.flip(frame, 1)

        H, W = frame.shape[:2]

        # Compose canvas with right dock (no obstruction)
        panel_w = 360
        canvas = np.zeros((H, W + panel_w, 3), dtype=np.uint8)
        canvas[:H, :W, :] = frame
        dock = (W, 0, W + panel_w, H)

        if paused:
            cv2.putText(canvas, "PAUSED", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3, cv2.LINE_AA)
            cv2.imshow("Arm Ability (Rehab)", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('p'): paused = False
            elif key == ord('q'): break
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hol.process(rgb)

        # Decide subject-side
        if side_pref == "auto":
            if res.right_hand_landmarks is not None: locked_side = "Right"
            elif res.left_hand_landmarks is not None: locked_side = "Left"
        else:
            locked_side = "Right" if side_pref == "Right" else "Left"

        # Hand landmarks for chosen subject-side
        hand_lm = None
        if locked_side == "Right" and res.right_hand_landmarks is not None:
            hand_lm = res.right_hand_landmarks
        elif locked_side == "Left" and res.left_hand_landmarks is not None:
            hand_lm = res.left_hand_landmarks

        pose_world = res.pose_world_landmarks.landmark if res.pose_world_landmarks is not None else None

        # Angles (raw)
        angles = {k: np.nan for k in ALL_KEYS}
        if locked_side is not None:
            angles["Elbow"]    = elbow_flexion_angle_world(pose_world, locked_side)
            angles["Shoulder"] = shoulder_abduction_angle_world(pose_world, locked_side)
            angles["Wrist"]    = wrist_flexion_angle(pose_world, hand_lm, locked_side, W, H)
            if hand_lm is not None:
                for f in FINGERS:
                    angles[f] = finger_ip_pip_angle(hand_lm, W, H, f)

        # Smooth raw angles first
        for k in angles:
            angles[k] = robust_add(angle_s[k], angles[k], SMOOTH_N_ANGLE)

        # Ability % using bounds (lo=min(N,M) -> 0, hi=max(N,M) -> 100)
        ability_raw = {k: np.nan for k in ALL_KEYS}  # unclamped
        ability     = {k: np.nan for k in ALL_KEYS}  # clamped 0..100 for UI
        for k, ang in angles.items():
            n = neutral.get(k, None); m = maxpos.get(k, None)
            pct = pct_from(ang, n, m)  # may be NaN, <0, >100
            ability_raw[k] = pct
            if np.isnan(pct):
                ability[k] = np.nan
            else:
                ability[k] = movavg(pct_s[k], clamp_percent(pct), SMOOTH_N_PCT)

        # Draw landmarks on the video part only (left)
        if res.pose_landmarks is not None:
            mp_draw.draw_landmarks(canvas[:, :W], res.pose_landmarks, mp_hol.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style())
        if res.left_hand_landmarks is not None:
            mp_draw.draw_landmarks(canvas[:, :W], res.left_hand_landmarks, mp_hol.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style())
        if res.right_hand_landmarks is not None:
            mp_draw.draw_landmarks(canvas[:, :W], res.right_hand_landmarks, mp_hol.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style())

        # ----- Right Dock UI -----
        x0, y0, x1, y1 = dock
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (30,30,30), -1)
        pad = 12; y = y0 + 28

        # Subject vs Screen label
        if locked_side is None:
            side_label = "--"
        else:
            if label_by_screen and selfie:
                side_label = "Left" if locked_side=="Right" else "Right"
            else:
                side_label = locked_side

        cv2.putText(canvas, f"Rehab HUD  [{side_label}]", (x0+pad, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.66, (255,255,255), 2, cv2.LINE_AA)
        y += 10

        # Session info
        elapsed = int(time.time() - start_time)
        mm, ss = divmod(elapsed, 60)
        y += 26; cv2.putText(canvas, f"Time  : {mm:02d}:{ss:02d}", (x0+pad, y), 0, 0.6, (220,220,220), 1, cv2.LINE_AA)
        y += 22; cv2.putText(canvas, f"Reps  : {session_reps}", (x0+pad, y), 0, 0.6, (220,220,220), 1, cv2.LINE_AA)

        # Bars
        def draw_bar(label, value_clamped, value_raw, yy, highlight=False, target=None):
            bar_w = panel_w - 2*pad; bar_h = 18
            x = x0 + pad; yb = yy

            # If not calibrated, tell user
            missing = (neutral.get(label, None) is None or maxpos.get(label, None) is None)
            if missing:
                cv2.putText(canvas, f"{label}: calibrate this metric", (x, yb-6), 0, 0.5, (80,200,255), 1, cv2.LINE_AA)
                cv2.rectangle(canvas, (x, yb), (x+bar_w, yb+bar_h), (70,70,70), 1)
                return

            cv2.rectangle(canvas, (x, yb), (x+bar_w, yb+bar_h), (70,70,70), 1)
            txt = "--"
            color = (120,200,120) if not np.isnan(value_clamped) else (80,80,80)
            if not np.isnan(value_clamped):
                txt = f"{value_clamped:5.1f}%"
                fill = int(bar_w * (value_clamped / 100.0))
                cv2.rectangle(canvas, (x+1, yb+1), (x+fill, yb+bar_h-1), color, -1)
                # target line
                if target is not None:
                    tx = x + int(bar_w * (target/100.0))
                    cv2.line(canvas, (tx, yb), (tx, yb+bar_h), (200,200,50), 2)
                    if abs(value_clamped - target) <= 5:
                        cv2.rectangle(canvas, (x, yb), (x+bar_w, yb+bar_h), (0,180,0), 2)

                # Out-of-range indicator (raw % <0 or >100)
                if not np.isnan(value_raw):
                    if value_raw < 0:
                        cv2.putText(canvas, "<", (x-10, yb+bar_h-3), 0, 0.5, (0, 180, 255), 1, cv2.LINE_AA)
                    elif value_raw > 100:
                        cv2.putText(canvas, ">", (x+bar_w+2, yb+bar_h-3), 0, 0.5, (0, 180, 255), 1, cv2.LINE_AA)

            col_label = (255,255,255) if highlight else (220,220,220)
            cv2.putText(canvas, f"{label}: {txt}", (x, yb-6),
                        0, 0.5, col_label, 1, cv2.LINE_AA)

        y += 30
        for k in JOINTS:
            draw_bar(k, ability[k], ability_raw[k], y, highlight=False, target=target_pct[k]); y += 26
        y += 8
        for k in FINGERS:
            draw_bar(k, ability[k], ability_raw[k], y, highlight=(k == current_finger_name()), target=target_pct[k]); y += 26

        y += 10
        cv2.putText(canvas, f"Selected finger: {current_finger_name()}   (use [ and ] to change)",
                    (x0+pad, y), 0, 0.5, (220,220,220), 1, cv2.LINE_AA)
        y += 18

        # Diagnostics (toggle with D)
        if show_diag:
            cv2.putText(canvas, "Diagnostics (ang / low / high / %raw)", (x0+pad, y), 0, 0.45, (200,200,255), 1, cv2.LINE_AA)
            y += 16
            def row_dbg(k):
                ang = angles.get(k, np.nan)
                n = neutral.get(k, None); m = maxpos.get(k, None)
                lo = None if (n is None or m is None) else min(n, m)
                hi = None if (n is None or m is None) else max(n, m)
                p = ability_raw.get(k, np.nan)
                txt = f"{k[:7]:>7}: {None if np.isnan(ang) else round(float(ang),2)} / {lo} / {hi} / {None if np.isnan(p) else round(float(p),2)}"
                cv2.putText(canvas, txt, (x0+pad, y), 0, 0.42, (180,180,180), 1, cv2.LINE_AA)
            for k in JOINTS:
                row_dbg(k); y += 14
            for k in FINGERS:
                row_dbg(k); y += 14

        cv2.putText(canvas, f"CSV: {'ON' if recording_csv else 'OFF'} | Video: {'REC' if recording_video else 'OFF'}",
                    (x0+pad, y), 0, 0.5, (200,255,200) if (recording_csv or recording_video) else (180,180,180), 1, cv2.LINE_AA)

        # ---- 10s Guided warm-up overlay ----
        if wizard_on:
            now = time.time()
            if wizard_countdown_end == 0:
                wizard_countdown_end = now + WARMUP_SECONDS
            remaining = int(max(0, wizard_countdown_end - now) + 0.999)

            step_name, keys, cue = wizard_steps[wizard_i]
            overlay = canvas[:, :W].copy()
            cv2.rectangle(overlay, (0, 0), (W, 140), (0, 0, 0), -1)
            alpha = 0.6
            canvas[:, :W] = cv2.addWeighted(overlay, alpha, canvas[:, :W], 1-alpha, 0)
            # Multiline cue
            lines = [f"{step_name}", *cue.split("\n"), f"Hold still... {remaining}s"]
            ytxt = 40
            for li, line in enumerate(lines):
                cv2.putText(canvas, line, (20, ytxt), 0, 0.9 if li==0 else 0.7,
                            (255,255,255), 2, cv2.LINE_AA)
                ytxt += 36

            if now >= wizard_countdown_end:
                # average burst for this step
                acc = burst_average(keys, bursts=12)
                is_neutral = (wizard_i in [0,2])
                for k in keys:
                    if acc[k] is None: 
                        continue
                    if is_neutral: neutral[k] = acc[k]
                    else:          maxpos[k]  = acc[k]
                wizard_i += 1
                if wizard_i >= len(wizard_steps):
                    wizard_on = False
                    wizard_i = 0
                    wizard_countdown_end = 0.0
                else:
                    wizard_countdown_end = now + WARMUP_SECONDS

        # CSV (log clamped UI values)
        if recording_csv:
            if csv_file is None:
                csv_file = CSV_PATH.open("a", buffering=1, encoding="utf-8")
                if csv_file.tell() == 0:
                    csv_file.write("timestamp," + ",".join([f"{k}_pct" for k in ALL_KEYS]) + "\n")
            row = [f"{ability[k]:.2f}" if not np.isnan(ability[k]) else "" for k in ALL_KEYS]
            csv_file.write(f"{time.time()}," + ",".join(row) + "\n")

        # Video
        if recording_video and writer is not None:
            writer.write(canvas)

        # Rep counting (simple hysteresis on selected metric)
        rep_metric = rep_metrics_cycle[rep_metric_i % len(rep_metrics_cycle)]
        cur = ability.get(rep_metric, np.nan)
        if not np.isnan(cur):
            if rep_state == 0 and cur >= up_thr:
                rep_state = 1
            elif rep_state == 1 and cur <= down_thr:
                rep_state = 0
                session_reps += 1

        # Show
        cv2.imshow("Arm Ability (Rehab)", canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('p'): paused = True
        elif key == ord('k'):
            png_path = OUT_DIR / f"snapshot_{time.strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(str(png_path), canvas); print(f"[SNAP] {png_path}")

        # Wizard toggle
        elif key == ord('w'):
            wizard_on = not wizard_on
            if wizard_on:
                wizard_i = 0; wizard_countdown_end = 0.0

        # --------- Grouped calibration (burst) ----------
        elif key == ord('1'):  # HAND Neutral
            vals = burst_average(["Wrist"] + FINGERS, bursts=12)
            for k in ["Wrist"] + FINGERS:
                if vals[k] is not None: neutral[k] = vals[k]
            print("[CALIB] HAND Neutral set.")
        elif key == ord('2'):  # HAND Max
            vals = burst_average(["Wrist"] + FINGERS, bursts=12)
            for k in ["Wrist"] + FINGERS:
                if vals[k] is not None: maxpos[k] = vals[k]
            print("[CALIB] HAND Max set.")
        elif key == ord('3'):  # ARM Neutral
            vals = burst_average(["Elbow","Shoulder"], bursts=12)
            for k in ["Elbow","Shoulder"]:
                if vals[k] is not None: neutral[k] = vals[k]
            print("[CALIB] ARM Neutral set.")
        elif key == ord('4'):  # ARM Max
            vals = burst_average(["Elbow","Shoulder"], bursts=12)
            for k in ["Elbow","Shoulder"]:
                if vals[k] is not None: maxpos[k] = vals[k]
            print("[CALIB] ARM Max set.")

        # --------- Per-finger selection & calibration ----------
        elif key == ord('['):  # prev finger
            sel_finger_i = (sel_finger_i - 1) % len(FINGERS)
            print(f"[FINGER] Selected: {current_finger_name()}")
        elif key == ord(']'):  # next finger
            sel_finger_i = (sel_finger_i + 1) % len(FINGERS)
            print(f"[FINGER] Selected: {current_finger_name()}")
        elif key == ord('5'):  # selected finger NEUTRAL
            fname = current_finger_name()
            vals = burst_average([fname], bursts=12)
            if vals[fname] is not None:
                neutral[fname] = vals[fname]
                print(f"[CALIB] {fname} Neutral set: {neutral[fname]:.2f}")
            else:
                print(f"[WARN] Could not set {fname} Neutral (landmarks not stable).")
        elif key == ord('6'):  # selected finger MAX
            fname = current_finger_name()
            vals = burst_average([fname], bursts=12)
            if vals[fname] is not None:
                maxpos[fname] = vals[fname]
                print(f"[CALIB] {fname} Max set: {maxpos[fname]:.2f}")
            else:
                print(f"[WARN] Could not set {fname} Max (landmarks not stable).")

        # --------- Save / targets / reps / logging ----------
        elif key == ord('n'):  # ALL Neutral (burst)
            vals = burst_average(ALL_KEYS, bursts=12)
            for k in ALL_KEYS:
                if vals[k] is not None: neutral[k] = vals[k]
            print("[CALIB] ALL Neutral set.")
        elif key == ord('m'):  # ALL Max (burst)
            vals = burst_average(ALL_KEYS, bursts=12)
            for k in ALL_KEYS:
                if vals[k] is not None: maxpos[k] = vals[k]
            print("[CALIB] ALL Max set.")
        elif key == ord('s'):
            save_calib({"neutral":neutral, "maxpos":maxpos}); print("[INFO] Saved arm_calibration.json")
        elif key == ord('r'):
            recording_csv = not recording_csv
            if not recording_csv and csv_file is not None:
                csv_file.close(); csv_file = None
                print(f"[INFO] Saved CSV to {CSV_PATH.resolve()}")
        elif key == ord('v'):
            recording_video = not recording_video
            if recording_video:
                ts = time.strftime("%Y%m%d_%H%M%S")
                out_path = OUT_DIR / f"arm_ability_{ts}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(out_path), fourcc, cam_fps, (W + panel_w, H))
                if not writer.isOpened():
                    print("[ERROR] Could not start video writer."); writer=None; recording_video=False
                else:
                    print(f"[REC] Video recording started: {out_path}")
            else:
                if writer is not None:
                    writer.release(); writer=None
                    print(f"[REC] Video saved to: {out_path.resolve()}")
                out_path = None
        elif key == ord('t'):
            target_idx = (target_idx + 1) % len(ALL_KEYS)
            print(f"[TARGET] Selected: {ALL_KEYS[target_idx]}")
        elif key == ord('y'):
            rep_metric_i = (rep_metric_i + 1) % len(rep_metrics_cycle)
            print(f"[REPS] Metric: {rep_metrics_cycle[rep_metric_i]}")
        elif key == ord('d'):
            show_diag = not show_diag
        elif key == 82:  # Up arrow
            k = ALL_KEYS[target_idx]
            cur = target_pct[k] or 0
            target_pct[k] = int(max(0, min(100, cur + 5)))
        elif key == 84:  # Down arrow
            k = ALL_KEYS[target_idx]
            cur = target_pct[k] or 0
            target_pct[k] = int(max(0, min(100, cur - 5)))

    if recording_csv and csv_file is not None:
        csv_file.close(); print(f"[INFO] Saved CSV to {CSV_PATH.resolve()}")
    if writer is not None:
        writer.release()
        if out_path: print(f"[REC] Video saved to: {out_path.resolve()}")
    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
