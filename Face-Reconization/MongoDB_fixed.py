# MongoDB Face Recognition Integration
# This script integrates face recognition with MongoDB for storing detection results

import os
import cv2
import imutils
import time
import pickle
import re
import numpy as np
from imutils.video import FPS
import datetime

# Get the script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Try to import pymongo, but make it optional
try:
    from pymongo import MongoClient
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    print("[WARNING] PyMongo not installed. MongoDB features will be disabled.")

# load serialized face detector
print("Loading Face Detector...")
protoPath = os.path.join(script_dir, "face_detection_model/deploy.prototxt")
modelPath = os.path.join(script_dir, "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel")
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load serialized face embedding model
print("Loading Face Recognizer...")
embedder = cv2.dnn.readNetFromTorch(os.path.join(script_dir, "Others/openface_nn4.small2.v1.t7"))

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open(os.path.join(script_dir, "output/recognizer"), "rb").read())
le = pickle.loads(open(os.path.join(script_dir, "output/le.pickle"), "rb").read())

# MongoDB Connection Setup (Optional)
db_connection = None
if MONGODB_AVAILABLE:
    try:
        # Update connection string if needed
        db_connection = MongoClient("mongodb://localhost:27017/")
        db = db_connection["face_recognition"]
        detections_collection = db["detections"]
        print("[INFO] Connected to MongoDB successfully")
    except Exception as e:
        print("[ERROR] Failed to connect to MongoDB:", str(e))
        MONGODB_AVAILABLE = False

# initialize the video stream
print("Starting Video Stream...")
vs = cv2.VideoCapture(0)
time.sleep(2.0)

# initialize the FPS counter
fps = FPS().start()

# Store recognized faces
recognized_faces = {}

print("Press 'q' to quit the application")

try:
    while True:
        # grab the frame from the video stream
        success, frame = vs.read()
        
        if not success:
            print("Failed to read from camera")
            break
        
        # resize the frame
        frame = imutils.resize(frame, width=600)
        (h, w) = frame.shape[:2]

        # construct a blob from the image
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300), 
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

        # apply face detector
        detector.setInput(imageBlob)
        detections = detector.forward()

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence
            confidence = detections[0, 0, i, 2]
            
            # filter out weak detections
            if confidence > 0.5:
                # compute the (x, y)-coordinates of the bounding box
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # extract the face ROI
                face = frame[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                # ensure the face width and height are sufficiently large
                if fW < 20 or fH < 20:
                    continue

                # construct a blob for the face ROI
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()

                # perform classification to recognize the face
                preds = recognizer.predict_proba(vec)[0]
                j = np.argmax(preds)
                proba = preds[j]
                name = le.classes_[j]
                
                # filter out weak predictions
                if proba < 0.5:
                    name = "Unknown"

                # Store or update detection time
                if name != "Unknown":
                    if name not in recognized_faces:
                        recognized_faces[name] = datetime.datetime.now()
                        
                        # Save to MongoDB if available
                        if MONGODB_AVAILABLE and db_connection:
                            try:
                                detection_data = {
                                    "name": name,
                                    "confidence": float(proba),
                                    "timestamp": datetime.datetime.now(),
                                    "status": "recognized"
                                }
                                detections_collection.insert_one(detection_data)
                                print(f"[MONGODB] Recorded detection for {name}")
                            except Exception as e:
                                print(f"[ERROR] Failed to save to MongoDB: {str(e)}")

                # draw the bounding box and name
                text = "{}: {:.2f}%".format(name, proba * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10

                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

        # update the FPS counter
        fps.update()

        # show the output frame
        cv2.imshow("Face Recognition - MongoDB Integration", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

except KeyboardInterrupt:
    print("\n[INFO] Interrupted by user")

finally:
    # stop the timer and display FPS information
    fps.stop()
    print("\nElapsed time: {:.2f}".format(fps.elapsed()))
    print("Approx. FPS: {:.2f}".format(fps.fps()))
    print(f"Total unique faces recognized: {len(recognized_faces)}")

    # cleanup
    cv2.destroyAllWindows()
    vs.release()
    
    if db_connection:
        db_connection.close()
        print("[INFO] MongoDB connection closed")
