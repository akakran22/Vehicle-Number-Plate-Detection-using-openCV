import cv2
import numpy as np
import imutils
import easyocr
import csv
import os

folder_path = "images"   #your folder path 
csv_filename = 'extracted_plates2.csv'

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Write CSV header
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Filename', 'Detected Numbers'])

# Iterate over each image file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(folder_path, filename)

        img = cv2.imread(image_path) #Loading the image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        bfilter = cv2.bilateralFilter(gray, 11, 17, 17)  # noise reduction
        edged = cv2.Canny(bfilter, 30, 200)  # Edge detection

        # Contours
        Keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(Keypoints)
        if contours:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
            location = None
            for contour in contours:
                approx = cv2.approxPolyDP(contour, 10, True)
                if len(approx) == 4:
                    location = approx
                    break
            if location is not None:
                mask = np.zeros(gray.shape, np.uint8)
                new_image = cv2.drawContours(mask, [location], 0, 255, -1)
                new_image = cv2.bitwise_and(img, img, mask=mask)

                (x, y) = np.where(mask == 255)
                (x1, y1) = (np.min(x), np.min(y))
                (x2, y2) = (np.max(x), np.max(y))
                cropped_image = gray[x1:x2 + 1, y1:y2 + 1]

                # Extract text using EasyOCR
                result = reader.readtext(cropped_image)
                if result:
                    text = result[0][-2]
                else:
                    text = "No text detected"

                # Write numbers to csv file
                with open(csv_filename, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([filename, text])
        else:
            print("No contours found in", filename)

print("Text along with file names has been written into the CSV file:", csv_filename)
