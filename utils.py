def get_main_object(objects):
    if len(objects) == 0:
        return None
    return objects[0]  # largest detected object


def filter_self_person(obj, frame_w, frame_h):
    name = obj["name"]

    # If not a person → keep it
    if name != "person":
        return True

    # Person detection → check if it's YOU in front camera
    x1, y1, x2, y2 = obj["box"]
    box_w = x2 - x1
    box_h = y2 - y1

    # Your own face occupies a big portion of the frame
    if box_w > 0.40 * frame_w or box_h > 0.40 * frame_h:
        return False  # ignore your face

    return True
