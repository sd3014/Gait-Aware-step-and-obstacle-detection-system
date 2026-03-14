import cv2
import numpy as np
from collections import deque

class TerrainDetector:
    def __init__(self):
        self.history = deque(maxlen=12)
        self.last_announced = None
        self.lock_frames = 0

    def analyze(self, depth, frame):
        depth = cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX)
        h, w = depth.shape

        # ---- Depth trend ----
        upper = depth[int(h*0.55):int(h*0.7), :]
        lower = depth[int(h*0.75):int(h*0.9), :]

        diff = np.mean(lower) - np.mean(upper)

        # ---- Conservative classification ----
        if diff > 0.12:
            label = "step_down"
        elif diff < -0.12:
            label = "step_up"
        else:
            label = "flat"

        self.history.append(label)

        return self._stable_decision()

    def _stable_decision(self):

        if self.lock_frames > 0:
            self.lock_frames -= 1
            return self.last_announced

        counts = {k: self.history.count(k) for k in set(self.history)}
        dominant = max(counts, key=counts.get)

        if counts[dominant] >= 4 and dominant != self.last_announced:
            self.last_announced = dominant
            self.lock_frames = 10
            return dominant

        return self.last_announced if self.last_announced else "safe"
