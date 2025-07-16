import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

# Initialize video capture and hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
folder = r"C:\Users\HP\OneDrive\Desktop\ASL_Interpreter\data\Z"
counter = 0

# Create directory if it doesn't exist
os.makedirs(folder, exist_ok=True)

# Variable to store the processed image
imgWhite = None
# Main loop
running = True
while running:
    success, img = cap.read()
    if not success:
        continue
    
    hands, img = detector.findHands(img)
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        
        # Calculate boundaries with safety checks
        imgHeight, imgWidth = img.shape[:2]
        y1, y2 = max(0, y - offset), min(imgHeight, y + h + offset)
        x1, x2 = max(0, x - offset), min(imgWidth, x + w + offset)
        
        imgCrop = img[y1:y2, x1:x2]
        
        if imgCrop.size > 0:
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            aspectRatio = h / w
            
            try:
                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize
                
                cv2.imshow("ImageCrop", imgCrop)
                cv2.imshow("ImageWhite", imgWhite)
                
            except Exception as e:
                print(f"Error processing image: {e}")
    
    cv2.imshow("Image", img)
    
    # Key controls - wait for key press for at least 1ms
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("s"):  # Press 's' to save current image
        if imgWhite is not None:
            timestamp = int(time.time())
            filename = f'{folder}/Image_{timestamp}_{counter}.jpg'
            cv2.imwrite(filename, imgWhite)
            print(f"Saved image {counter} at {filename}")
            counter += 1
        else:
            print("No hand detected to save!")
    elif key == ord("q") or key == 27:  # Press 'q' or ESC to quit
        running = False
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print(f"Program ended. Total images saved: {counter}")    