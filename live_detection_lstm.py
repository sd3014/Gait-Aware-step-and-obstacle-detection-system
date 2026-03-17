import cv2
import mediapipe as mp
import numpy as np
import threading, time
from collections import deque
from tensorflow.keras.models import load_model

# ── Load clean model ──────────────────────────────────────────────────────────
model = load_model("model_clean.keras", compile=False)
print("✅ Model ready:", model.input_shape)

SEQ_LEN      = 10
NUM_FEATURES = 69

# ── MediaPipe ─────────────────────────────────────────────────────────────────
mp_pose = mp.solutions.pose
pose    = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# ── Colored skeleton ──────────────────────────────────────────────────────────
CONN = [
    (0,1,(255,200,0)),(1,2,(255,200,0)),(2,3,(255,200,0)),(3,7,(255,200,0)),
    (0,4,(255,200,0)),(4,5,(255,200,0)),(5,6,(255,200,0)),(6,8,(255,200,0)),
    (11,12,(0,255,180)),(11,23,(0,255,180)),(12,24,(0,255,180)),(23,24,(0,255,180)),
    (11,13,(255,120,0)),(13,15,(255,80,0)),(15,17,(255,60,0)),(15,19,(255,60,0)),
    (12,14,(0,120,255)),(14,16,(0,80,255)),(16,18,(0,60,255)),(16,20,(0,60,255)),
    (23,25,(180,255,0)),(25,27,(140,255,0)),(27,29,(100,230,0)),(27,31,(100,230,0)),
    (24,26,(0,220,180)),(26,28,(0,180,160)),(28,30,(0,160,140)),(28,32,(0,160,140)),
]

def draw_skeleton(frame, lms, fall=False):
    h, w = frame.shape[:2]
    pts = {i: (int(lm.x*w), int(lm.y*h)) for i,lm in enumerate(lms.landmark)}
    for s,e,col in CONN:
        if s in pts and e in pts:
            cv2.line(frame, pts[s], pts[e], (0,0,255) if fall else col, 2, cv2.LINE_AA)
    for cx,cy in pts.values():
        cv2.circle(frame, (cx,cy), 4, (0,0,255) if fall else (255,255,255), -1, cv2.LINE_AA)
        cv2.circle(frame, (cx,cy), 4, (0,0,0), 1, cv2.LINE_AA)

# ── Voice alert ───────────────────────────────────────────────────────────────
_t = 0
def speak():
    global _t
    if time.time()-_t < 6: return
    _t = time.time()
    def _r():
        try:
            import pyttsx3
            e = pyttsx3.init(); e.setProperty("rate",155)
            e.say("Fall detected! Please check immediately.")
            e.runAndWait()
        except: pass
    threading.Thread(target=_r, daemon=True).start()

# ── Main loop ─────────────────────────────────────────────────────────────────
buf       = deque(maxlen=SEQ_LEN)
status    = "Warming up..."
txt_color = (255,255,255)
fall_flag = False

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print("🎥 Live — press ESC to quit")

while True:
    ret, frame = cap.read()
    if not ret: continue

    frame   = cv2.flip(frame, 1)
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.pose_landmarks:
        lms = results.pose_landmarks
        draw_skeleton(frame, lms, fall=fall_flag)

        feats = []
        for lm in lms.landmark:
            feats += [lm.x, lm.y]
        feats = (feats + [0.0]*NUM_FEATURES)[:NUM_FEATURES]
        buf.append(feats)

        if len(buf) == SEQ_LEN:
            prob      = float(model.predict(np.array(buf).reshape(1,SEQ_LEN,NUM_FEATURES), verbose=0)[0][0])
            fall_flag = prob > 0.5
            status    = f"FALL  {prob:.0%}" if fall_flag else f"Normal  {1-prob:.0%}"
            txt_color = (0,0,255) if fall_flag else (0,220,80)
            if fall_flag: speak()

    # HUD
    W,H = frame.shape[1], frame.shape[0]
    ov  = frame.copy()
    cv2.rectangle(ov,(0,0),(W,55),(0,0,0),-1)
    cv2.addWeighted(ov,0.5,frame,0.5,0,frame)
    cv2.putText(frame, status, (20,38), cv2.FONT_HERSHEY_DUPLEX, 1.0, txt_color, 2, cv2.LINE_AA)

    bw = int(len(buf)/SEQ_LEN*160)
    cv2.rectangle(frame,(W-180,15),(W-20,38),(50,50,50),-1)
    cv2.rectangle(frame,(W-180,15),(W-180+bw,38),(80,180,80),-1)
    cv2.putText(frame,f"{len(buf)}/{SEQ_LEN}",(W-175,34),cv2.FONT_HERSHEY_SIMPLEX,0.45,(200,200,200),1)

    if fall_flag:
        cv2.rectangle(frame,(0,H-60),(W,H),(0,0,180),-1)
        cv2.putText(frame,"!! FALL DETECTED !!",(W//2-175,H-15),
                    cv2.FONT_HERSHEY_DUPLEX,1.1,(255,255,255),2,cv2.LINE_AA)

    cv2.imshow("Gait Fall Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()
pose.close()