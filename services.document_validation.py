import cv2


def validate_document(image_path):

    img = cv2.imread(image_path)

    if img is None:
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    brightness = gray.mean()

    if brightness < 50:
        return False

    return True