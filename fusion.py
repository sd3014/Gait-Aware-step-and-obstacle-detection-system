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

def fuse_decisions(terrain, obj_name, direction):

    # ---- Highest Priority ----
    if terrain == "steep_drop":
        return "Danger. Drop ahead. Stop."

    if terrain == "step_down":
        if obj_name and direction == "center":
            # Furniture blocking view, likely not stair
            return f"{obj_name} ahead."

        return "Danger. Step down ahead."


    if terrain == "step_up":
        return "Step up ahead."

    # ---- Object Priority ----
    if obj_name:
        if direction == "center":
            return f"{obj_name} ahead."
        else:
            return f"{obj_name} on the {direction}."

    # ---- Slopes ----
    if terrain == "mild_up":
        return "Gentle upward slope."

    if terrain == "mild_down":
        return "Gentle downward slope."

    # ---- Safe ----
    if terrain == "safe":
        return "Clear path."

    return None
