# app.py
import os, time, json, cv2, numpy as np, gradio as gr, mediapipe as mp
from pathlib import Path
from ability_core import (
    Smoother, pct_from_bounds, clamp_percent,
    finger_ip_pip_angle, wrist_flexion_angle,
    elbow_flexion_angle_world, shoulder_abduction_angle_world,
)

APP_VERSION = "cloud-1.0"
OUT_DIR = Path("outputs"); OUT_DIR.mkdir(exist_ok=True, parents=True)
CALIB_PATH = Path("arm_calibration.json")

FINGERS = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
JOINTS  = ["Wrist", "Elbow", "Shoulder"]
ALL_KEYS = JOINTS + FINGERS

mp_hol = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

def new_state():
    return dict(
        side_pref="auto", selfie=True, label_by_screen=True,
        neutral={k: None for k in ALL_KEYS},
        maxpos={k: None for k in ALL_KEYS},
        smooth_angle={k: Smoother(5) for k in ALL_KEYS},
        smooth_pct={k: Smoother(5) for k in ALL_KEYS},
        right_dock_w=360, start_ts=time.time(),
        selected_finger=0, target_pct={k: None for k in ALL_KEYS},
    )

def save_calib(neutral, maxpos):
    CALIB_PATH.write_text(json.dumps({"neutral":neutral, "maxpos":maxpos}, indent=2))

def burst_average(hol, frame, keys, side_pref, selfie):
    """Use current frame to compute requested raw angles."""
    H, W = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hol.process(rgb)
    # decide side
    if side_pref == "auto":
        side = "Right" if res.right_hand_landmarks is not None else ("Left" if res.left_hand_landmarks is not None else None)
    else:
        side = "Right" if side_pref == "right" else "Left"
    hand = None
    if side == "Right" and res.right_hand_landmarks is not None: hand = res.right_hand_landmarks
    if side == "Left"  and res.left_hand_landmarks  is not None: hand = res.left_hand_landmarks
    pw = res.pose_world_landmarks.landmark if res.pose_world_landmarks is not None else None

    vals = {}
    if side is not None:
        vals["Elbow"]    = elbow_flexion_angle_world(pw, side)
        vals["Shoulder"] = shoulder_abduction_angle_world(pw, side)
        vals["Wrist"]    = wrist_flexion_angle(pw, hand, side, W, H)
    if hand is not None:
        for f in FINGERS:
            vals[f] = finger_ip_pip_angle(hand, W, H, f)
    return {k: (None if k not in vals or np.isnan(vals[k]) else float(vals[k])) for k in keys}

def process_frame(frame, state):
    if frame is None:
        return None, state
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if state["selfie"]:
        frame = cv2.flip(frame, 1)
    H, W = frame.shape[:2]

    # canvas with right dock
    panel_w = state["right_dock_w"]
    canvas = np.zeros((H, W + panel_w, 3), dtype=np.uint8)
    canvas[:H, :W] = frame
    x0, y0, x1, y1 = W, 0, W + panel_w, H
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (30,30,30), -1)

    hol = mp_hol.Holistic(static_image_mode=False, model_complexity=1,
                            refine_face_landmarks=False, min_detection_confidence=0.6,
                            min_tracking_confidence=0.6)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hol.process(rgb)

    # side (subject)
    if state["side_pref"] == "auto":
        side = "Right" if res.right_hand_landmarks is not None else ("Left" if res.left_hand_landmarks is not None else None)
    else:
        side = "Right" if state["side_pref"] == "right" else "Left"

    # draw landmarks
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

    # raw angles
    angles = {k: np.nan for k in ALL_KEYS}
    hand = None
    if side == "Right" and res.right_hand_landmarks is not None: hand = res.right_hand_landmarks
    if side == "Left"  and res.left_hand_landmarks  is not None: hand = res.left_hand_landmarks
    pw = res.pose_world_landmarks.landmark if res.pose_world_landmarks is not None else None

    if side is not None:
        angles["Elbow"]    = elbow_flexion_angle_world(pw, side)
        angles["Shoulder"] = shoulder_abduction_angle_world(pw, side)
        angles["Wrist"]    = wrist_flexion_angle(pw, hand, side, W, H)
    if hand is not None:
        for f in FINGERS:
            angles[f] = finger_ip_pip_angle(hand, W, H, f)

    # smooth + map to %
    ability = {}; ability_raw = {}
    for k, ang in angles.items():
        ang_s = state["smooth_angle"][k].add(ang)
        n = state["neutral"][k]; m = state["maxpos"][k]
        p_raw = pct_from_bounds(ang_s, n, m)
        ability_raw[k] = p_raw
        ability[k] = state["smooth_pct"][k].add(clamp_percent(p_raw) if not np.isnan(p_raw) else np.nan)

    # HUD
    pad = 12; y = y0 + 28
    # label
    if side is None: side_label = "--"
    else:
        side_label = ("Left" if side == "Right" else "Right") if (state["label_by_screen"] and state["selfie"]) else side
    cv2.putText(canvas, f"Rehab HUD [{side_label}] v{APP_VERSION}", (x0+pad, y), 0, 0.6, (255,255,255), 2, cv2.LINE_AA)

    y += 34
    def draw_bar(label, v_clamped, v_raw, yy, target=None):
        bar_w = panel_w - 2*pad; bar_h = 18
        x = x0 + pad; yb = yy
        if state["neutral"][label] is None or state["maxpos"][label] is None:
            cv2.putText(canvas, f"{label}: calibrate", (x, yb-6), 0, 0.5, (80,200,255), 1, cv2.LINE_AA)
            cv2.rectangle(canvas, (x, yb), (x+bar_w, yb+bar_h), (70,70,70), 1)
            return
        cv2.rectangle(canvas, (x, yb), (x+bar_w, yb+bar_h), (70,70,70), 1)
        txt = "--"
        if not np.isnan(v_clamped):
            txt = f"{v_clamped:5.1f}%"
            fill = int(bar_w * (v_clamped / 100.0))
            cv2.rectangle(canvas, (x+1, yb+1), (x+fill, yb+bar_h-1), (120,200,120), -1)
            if target is not None:
                tx = x + int(bar_w * (target/100.0))
                cv2.line(canvas, (tx, yb), (tx, yb+bar_h), (200,200,50), 2)
            # out-of-range indicators
            if not np.isnan(v_raw):
                if v_raw < 0:  cv2.putText(canvas, "<", (x-10, yb+bar_h-3), 0, 0.5, (0,180,255), 1, cv2.LINE_AA)
                if v_raw > 100:cv2.putText(canvas, ">", (x+bar_w+2, yb+bar_h-3), 0, 0.5, (0,180,255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{label}: {txt}", (x, yb-6), 0, 0.5, (220,220,220), 1, cv2.LINE_AA)

    for k in JOINTS:
        draw_bar(k, ability[k], ability_raw[k], y, target=state["target_pct"][k]); y += 26
    y += 8
    for k in FINGERS:
        draw_bar(k, ability[k], ability_raw[k], y, target=state["target_pct"][k]); y += 26

    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), state

# ---- UI actions
def ui_calibrate(action, state, frame):
    hol = mp_hol.Holistic(static_image_mode=False, model_complexity=1,
                            refine_face_landmarks=False, min_detection_confidence=0.6,
                            min_tracking_confidence=0.6)
    if action == "hand_neutral":
        vals = burst_average(hol, frame, ["Wrist"] + FINGERS, state["side_pref"], state["selfie"])
        for k in ["Wrist"] + FINGERS:
            if vals[k] is not None: state["neutral"][k] = vals[k]
    elif action == "hand_max":
        vals = burst_average(hol, frame, ["Wrist"] + FINGERS, state["side_pref"], state["selfie"])
        for k in ["Wrist"] + FINGERS:
            if vals[k] is not None: state["maxpos"][k] = vals[k]
    elif action == "arm_neutral":
        vals = burst_average(hol, frame, ["Elbow","Shoulder"], state["side_pref"], state["selfie"])
        for k in ["Elbow","Shoulder"]:
            if vals[k] is not None: state["neutral"][k] = vals[k]
    elif action == "arm_max":
        vals = burst_average(hol, frame, ["Elbow","Shoulder"], state["side_pref"], state["selfie"])
        for k in ["Elbow","Shoulder"]:
            if vals[k] is not None: state["maxpos"][k] = vals[k]
    elif action == "save":
        CALIB_PATH.write_text(json.dumps({"neutral":state["neutral"], "maxpos":state["maxpos"]}, indent=2))
    return state

def ui_select_finger(delta, state):
    n = len(FINGERS)
    idx = (state.get("selected_finger", 0) + delta) % n
    state["selected_finger"] = idx
    return state, f"Selected: {FINGERS[idx]}"

def ui_toggle_labelmode(mode, state):
    state["label_by_screen"] = (mode == "screen")
    return state

def ui_toggle_selfie(v, state):
    state["selfie"] = v
    return state

def build_app():
    with gr.Blocks(title=f"Rehab Arm Ability v{APP_VERSION}") as demo:
        gr.Markdown(f"### Rehab Arm Ability (Cloud) — v{APP_VERSION}\nCalibrate bounds (Neutral/Max) → HUD shows 0–100% with out-of-range markers.")
        state = gr.State(new_state())
        cam = gr.Image(sources=["webcam"], streaming=True, label="Webcam", height=480)
        out = gr.Image(label="HUD", height=480)

        with gr.Row():
            gr.Markdown("**Calibration (Group):**")
            btn_hn = gr.Button("Hand Neutral"); btn_hm = gr.Button("Hand Max")
            btn_an = gr.Button("Arm Neutral");  btn_am = gr.Button("Arm Max")
            btn_sv = gr.Button("Save Calibration")

        with gr.Row():
            gr.Markdown("**Per-Finger:**")
            btn_prev = gr.Button("◀ Prev Finger")
            txt_sel  = gr.Textbox(value="Selected: Thumb", interactive=False)
            btn_next = gr.Button("Next Finger ▶")

        with gr.Row():
            mode = gr.Radio(["subject","screen"], value="screen", label="Label Left/Right by")
            selfie = gr.Checkbox(value=True, label="Selfie preview")

        # stream loop
        def stream(frame, state):
            img, state = process_frame(frame, state)
            return img, state
        cam.stream(stream, inputs=[cam, state], outputs=[out, state])

        # wires
        btn_hn.click(ui_calibrate, inputs=[gr.Textbox(value="hand_neutral", visible=False), state, cam], outputs=[state])
        btn_hm.click(ui_calibrate, inputs=[gr.Textbox(value="hand_max", visible=False), state, cam], outputs=[state])
        btn_an.click(ui_calibrate, inputs=[gr.Textbox(value="arm_neutral", visible=False), state, cam], outputs=[state])
        btn_am.click(ui_calibrate, inputs=[gr.Textbox(value="arm_max", visible=False), state, cam], outputs=[state])
        btn_sv.click(ui_calibrate, inputs=[gr.Textbox(value="save", visible=False), state, cam], outputs=[state])

        btn_prev.click(ui_select_finger, inputs=[gr.Number(value=-1, visible=False), state], outputs=[state, txt_sel])
        btn_next.click(ui_select_finger, inputs=[gr.Number(value=1, visible=False), state], outputs=[state, txt_sel])

        mode.change(ui_toggle_labelmode, inputs=[mode, state], outputs=[state])
        selfie.change(ui_toggle_selfie, inputs=[selfie, state], outputs=[state])

    return demo

if __name__ == "__main__":
    demo = build_app()
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
