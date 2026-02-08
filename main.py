import cv2
from depth_model import get_depth
from object_model import detect_objects
from fusion import get_direction, get_terrain_from_depth
from logic import decide_message
from cooldown import is_cooldown_over
from voice import speak
from utils import get_main_object, filter_self_person

cap = cv2.VideoCapture(1)   # DroidCam camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # ----- TERRAIN DETECTION (MiDaS) -----
    depth = get_depth(frame)
    terrain = get_terrain_from_depth(depth, frame)
    print("TERRAIN:", terrain)

    cv2.putText(
        frame,
        f"TERRAIN: {terrain}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    # ----- OBJECT DETECTION -----
    objects = detect_objects(frame)

    # Filter your own face (important for testing)
    objects = [o for o in objects if filter_self_person(o, w, h)]

    main_obj = get_main_object(objects)

    direction = None
    obj_name = None

    if main_obj:
        direction = get_direction(main_obj["box"], w)
        obj_name = main_obj["name"]

        x1, y1, x2, y2 = main_obj["box"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"{obj_name} ({direction})"
        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # ----- VOICE GUIDANCE -----
    message = decide_message(terrain, obj_name, direction)

    if message and is_cooldown_over():
        speak(message)

    # ----- GUI -----
    cv2.imshow("Assistive Navigation - Mobile Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
