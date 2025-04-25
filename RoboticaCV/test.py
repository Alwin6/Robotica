import cv2 as cv
import numpy as np
import math as m

cv.namedWindow("Strawberry Detection")
cap = cv.VideoCapture('rtsp://admin:admin@192.168.42.1:554/live')

if cap.isOpened(): # try to get the first frame
    rval, frame = cap.read()
else:
    rval = False

values = []

while rval:
    ret, img = cap.read()
    if not ret:
        break

    # Convert to HSV for better color segmentation
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # Define red color range (strawberries are usually red)
    lower_red = np.array([170, 100, 100])
    upper_red = np.array([180, 255, 255])

    # Create masks and combine
    mask = cv.inRange(hsv, lower_red, upper_red)

    # Morphological operations to clean up noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    # Find contours in the mask
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Filter and draw contours
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > 3000:  # filter small objects
            x, y, w, h = cv.boundingRect(cnt)
            cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.putText(img, f'Strawberry {round(5121.2 * area ** -0.479)}', (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show result
    cv.imshow('Strawberry Detection', img)

    # Clear with 'c'
    if cv.waitKey(1) & 0xFF == ord('c'):
        values = []

    # Quit with 'q'
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv.destroyWindow("Strawberry Detection")
cap.release()