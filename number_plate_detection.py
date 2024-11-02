import cv2
import imutils
import numpy as np

def process_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (600, 400))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 13, 15, 15)

    edged = cv2.Canny(gray, 30, 200)

    contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    count = None

    for c in contours:
        perimeter = cv2.arcLength(c, True)
        points = cv2.approxPolyDP(c, 0.018 * perimeter, True)

        if len(points) == 4:
            count = points
            break

    if count is not None:
        cv2.drawContours(img, [count], -1, (0, 255, 0), 3) 
    else:
        print(f"No number plate detected in {image_path}")

    cv2.imshow('Number Plate', img)

for i in range(1, 10):  
    image_path = f'ANPR Test/Cars{i}.png'  
    process_image(image_path)
    cv2.waitKey(0) 

cv2.destroyAllWindows()