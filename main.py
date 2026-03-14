import cv2
from depth_model import get_depth
from object_model import detect_objects
from fusion import get_direction, fuse_decisions
from terrain_detector import TerrainDetector
from cooldown import is_cooldown_over
from voice import speak
from utils import get_main_object, filter_self_person
import time


cap = cv2.VideoCapture(1)

terrain_detector = TerrainDetector()

last_speak_time = 0
SPEAK_INTERVAL = 4 

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()

    h, w, _ = frame.shape

    # ----- TERRAIN (STABLE) -----
    depth = get_depth(frame)
    terrain = terrain_detector.analyze(depth, frame)

    terrain_for_fusion = terrain if terrain else "safe"
    display_terrain = terrain_for_fusion

    cv2.putText(
        frame,
        f"TERRAIN: {display_terrain}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    # ----- OBJECT DETECTION -----
    objects = detect_objects(frame)
    objects = [o for o in objects if filter_self_person(o, w, h)]
    main_obj = get_main_object(objects)

    direction = None
    obj_name = None

    if main_obj:
        direction = get_direction(main_obj["box"], w)
        obj_name = main_obj["name"]

        x1, y1, x2, y2 = main_obj["box"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{obj_name} ({direction})",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # ----- FUSION (SINGLE DECISION POINT) -----
    message = fuse_decisions(terrain_for_fusion, obj_name, direction)

    print("MESSAGE:", message)

    if message:
        if time.time() - last_speak_time >= 4:
            speak(message)
            last_speak_time = time.time()

    # ----- GUI -----
    cv2.imshow("Assistive Navigation - Mobile Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
