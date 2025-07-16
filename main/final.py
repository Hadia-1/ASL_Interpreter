import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
offset = 20
imgSize = 300
labels = ["_", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", 
          "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

sentence = ""
capture_mode = False

try:
    while True:
        success, img = cap.read()
        if not success:
            print("Warning: Frame not read properly")
            continue  # Skip this iteration instead of breaking
            
        imgOutput = img.copy()
        
        # Sentence bar
        bar_height = 70
        cv2.rectangle(imgOutput, (0, img.shape[0] - bar_height), 
                     (img.shape[1], img.shape[0]), (0, 0, 0), cv2.FILLED)
        cv2.putText(imgOutput, f"Sentence: {sentence}", (10, img.shape[0] - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        try:
            hands, img = detector.findHands(img)
            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']
                
                # Ensure crop coordinates are within image bounds
                y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
                x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)
                imgCrop = img[y1:y2, x1:x2]
                
                if imgCrop.size == 0:
                    raise ValueError("Empty crop region")
                
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
                    
                    prediction, index = classifier.getPrediction(imgWhite, draw=False)
                    
                    if capture_mode and index != 0:
                        sentence += labels[index]
                        capture_mode = False
                    
                    # Visual feedback
                    cv2.rectangle(imgOutput, (x - offset, y - offset-50),
                                  (x - offset+90, y - offset-50+50), (255, 0, 255), cv2.FILLED)
                    cv2.putText(imgOutput, labels[index], (x, y -26), 
                               cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
                    cv2.rectangle(imgOutput, (x-offset, y-offset),
                                  (x + w+offset, y + h+offset), (255, 0, 255), 4)
                    
                    cv2.imshow("ImageCrop", imgCrop)
                    cv2.imshow("ImageWhite", imgWhite)
                
                except Exception as e:
                    print(f"Processing error: {e}")
                    continue

        except Exception as e:
            print(f"Hand detection error: {e}")
            continue
        
        cv2.imshow("Image", imgOutput)
        
        key = cv2.waitKey(10) & 0xFF
        if key == ord('c'):
            capture_mode = True
        elif key == ord(' '):
            sentence += " "
        elif key == 8:  # Backspace
            sentence = sentence[:-1]
        elif key == 27 or key == ord('q'):
            print("Final Sentence:", sentence)
            break

except KeyboardInterrupt:
    print("\nProgram interrupted by user")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Resources released")