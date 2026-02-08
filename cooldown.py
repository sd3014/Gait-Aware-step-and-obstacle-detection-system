import time

last_message_time = 0
cooldown_duration = 2  # seconds

def is_cooldown_over():
    global last_message_time
    now = time.time()
    if now - last_message_time >= cooldown_duration:
        last_message_time = now
        return True
    return False
