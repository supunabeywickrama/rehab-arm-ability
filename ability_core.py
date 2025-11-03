# ability_core.py
import math, numpy as np
from collections import deque

VIS_THRESH = 0.50

POSE_IDX = {
    "Left":  {"shoulder":11, "elbow":13, "wrist":15, "hip":23},
    "Right": {"shoulder":12, "elbow":14, "wrist":16, "hip":24},
}

def clamp_percent(p): 
    return max(0.0, min(100.0, float(p)))

def angle_between(v1, v2):
    v1 = np.asarray(v1, np.float32); v2 = np.asarray(v2, np.float32)
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0: return np.nan
    c = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(math.degrees(math.acos(c)))

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

def pct_from_bounds(ang, n, m):
    """
    Direction-agnostic bounds mapping:
      low=min(n,m)->0%, high=max(n,m)->100%. Returns unclamped % (may be <0 or >100).
    """
    if ang is None or n is None or m is None: return np.nan
    if any(map(np.isnan, [ang, n, m])): return np.nan
    lo, hi = (n, m) if n <= m else (m, n)
    denom = hi - lo
    if denom < 1e-3: return np.nan
    return 100.0 * ((ang - lo) / denom)

class Smoother:
    def __init__(self, n=5): 
        self.buf = deque(maxlen=n)
    def add(self, x):
        if not (x is None or np.isnan(x)): 
            self.buf.append(float(x))
        return float(np.mean(self.buf)) if self.buf else np.nan
