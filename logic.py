import time
from voice import turning_instruction   # <-- IMPORTANT (import turning function)

# Cooldown timers
last_warning = 0
COOLDOWN = 2   # seconds

def should_talk():
    global last_warning
    now = time.time()
    if now - last_warning > COOLDOWN:
        last_warning = now
        return True
    return False


def decide_message(terrain, obj_name, direction):

    if terrain == "step up":
        return "Step up ahead."

    if terrain == "step down or pit":
        return "Caution. Step down ahead."

    if terrain == "uneven surface":
        return "Uneven ground ahead."

    if terrain == "flat ground" and obj_name is None:
        return "Clear path."

    if obj_name:
        if direction == "center":
            return f"{obj_name} ahead."
        return f"{obj_name} on the {direction}."

    return None
