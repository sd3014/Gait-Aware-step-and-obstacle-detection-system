import cv2
import numpy as np
def get_direction(box, frame_width):
    x1, y1, x2, y2 = box
    obj_center = (x1 + x2) // 2
    center = frame_width // 2

    offset = obj_center - center  # negative = left, positive = right
    ratio = offset / center  # normalized (-1 to 1)

    if ratio < -0.75:
        return "far left"
    elif ratio < -0.45:
        return "left"
    elif ratio < -0.15:
        return "slight left"
    elif ratio < 0.15:
        return "center"
    elif ratio < 0.45:
        return "slight right"
    elif ratio < 0.75:
        return "right"
    else:
        return "far right"

def get_terrain_from_depth(depth, frame):
    """
    Correctly distinguish step UP vs step DOWN using:
    1. Relative depth trend (MiDaS)
    2. Vertical position of stair edge
    """

    depth = cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX)
    h, w = depth.shape

    # -------- DEPTH TREND --------
    upper = depth[int(h*0.55):int(h*0.7), :]
    lower = depth[int(h*0.75):int(h*0.9), :]

    upper_mean = np.mean(upper)
    lower_mean = np.mean(lower)
    diff = lower_mean - upper_mean

    # -------- EDGE POSITION --------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Sum edges row-wise to find strongest horizontal edge
    edge_strength = np.sum(edges, axis=1)
    strongest_edge_row = np.argmax(edge_strength)

    edge_position_ratio = strongest_edge_row / h

    print(
        "DEPTH → upper:", round(upper_mean,3),
        "lower:", round(lower_mean,3),
        "diff:", round(diff,3),
        "| EDGE ROW:", strongest_edge_row,
        "ratio:", round(edge_position_ratio,2)
    )

    # -------- FINAL DECISION --------

    # Significant depth change exists
    if abs(diff) > 0.08:

        # Stair edge appears HIGH → step UP
        if edge_position_ratio < 0.55:
            return "step up"

        # Stair edge appears LOW → step DOWN / pit
        else:
            return "step down or pit"

    # Mild variation
    elif abs(diff) > 0.04:
        return "uneven surface"

    return "flat ground"