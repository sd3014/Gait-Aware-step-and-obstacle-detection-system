import pyttsx3

engine = pyttsx3.init()

def speak(text):
    if text:
        engine.say(text)
        engine.runAndWait()


def turning_instruction(direction, obj_name):
    if direction == "far left":
        return f"{obj_name} far left, turn completely right."

    if direction == "left":
        return f"{obj_name} on the left, turn more right."

    if direction == "slight left":
        return f"{obj_name} slightly on the left, turn slightly right."

    if direction == "center":
        return f"{obj_name} ahead, move right."

    if direction == "slight right":
        return f"{obj_name} slightly on the right, turn slightly left."

    if direction == "right":
        return f"{obj_name} on the right, turn more left."

    if direction == "far right":
        return f"{obj_name} far right, turn completely left."

    return None


def give_guidance(step, direction, obj_name):
    # STEP WARNINGS (Priority 1)
    if step == "very high step":
        speak("Danger. Very high step ahead. Stop.")
        return

    if step == "high step":
        speak("High step ahead. Be careful.")
        return

    if step == "medium step":
        speak("Medium step ahead.")
        return

    if step == "small step":
        speak("Small step ahead.")
        return

    # OBSTACLE TURNING INSTRUCTION (Priority 2)
    if obj_name and direction:
        turn_msg = turning_instruction(direction, obj_name)
        speak(turn_msg)
        return

    # CLEAR PATH (Priority 3)
    if step == "flat ground" and obj_name is None:
        speak("Path clear.")
        return
