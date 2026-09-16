import cv2
import numpy as np

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("camera open failed")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("frame read failed")
        break

    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    img = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    img = img.astype(np.uint8)

    color = cv2.applyColorMap(img, cv2.COLORMAP_INFERNO)

    cv2.imshow("FLIR Thermal", color)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()