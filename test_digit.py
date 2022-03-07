import cv2
from process_image import get_output_image

cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    #frame_copy = frame.copy()
    output_img = get_output_image(frame)
    cv2.imshow('Detected text', output_img)

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()