from ultralytics import YOLO
import cv2
import cvzone
import time
import math
from sort import *
FILE_PATH = "Videos/traffic.mp4"

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]
# Todo - Add Mask To Reduce Compute Resources

# For Tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
cap  = cv2.VideoCapture(FILE_PATH)
model = YOLO("yolo_weights/yolov8l.pt") 

# Initialize the coordinates for line which is specific for the video
limitsUp = [334, 121, 544, 111]
limitsDown = [286, 247, 647, 235]

# initialize variables
start_time = 0
end_time = 0
last_centroids = {}
current_centroids = {}
starter = {}
ender = {}

while True:
    _, img = cap.read()
    if not _:
        break

    results = model(img, stream=True)
    detections = np.empty((0,5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)


            # Classname
            cls = int(box.cls[0])

            # Confodence score
            conf = math.ceil(box.conf[0]*100)/100
            if conf > 0.5:
                cvzone.putTextRect(img, f'{classNames[cls]}', (x2,y2), scale=1, thickness=1, colorR=(0,0,255))
                currentArray = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections, currentArray))

    resultTracker = tracker.update(detections)     

    cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 0, 255), 5)
    cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 0, 255), 5)

    for res in resultTracker:
        x1,y1,x2,y2,id = res
        x1,y1,x2,y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
        w,h = x2-x1, y2-y1

        cvzone.putTextRect(img, f'ID: {id}', (x1,y1), scale=1, thickness=1, colorR=(0,0,255))
        cvzone.cornerRect(img, (x1,y1,w,h), l=9, rt=1, colorR=(255,0,255))

        cx, cy = x1 + w // 2, y1 + h // 2
        centroid = (cx, cy)
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)   

        if limitsUp[0] < cx < limitsUp[2] and limitsUp[1] - 5 < cy < limitsUp[1] + 5:
            start_time = time.time()
            starter[id] = int(start_time)
            last_centroids[id] = (cx, cy)
            cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 255, 0), 5)

        if limitsDown[0] < cx < limitsDown[2] and limitsDown[1] - 15 < cy < limitsDown[1] + 15:
            if id in last_centroids:
                end_time = time.time()
                ender[id] = int(end_time)
                current_centroids[id] = (cx, cy)
                cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 255, 0), 5)

        if id in current_centroids:
            distance = ((current_centroids[id][0] - last_centroids[id][0])**2 + (current_centroids[id][1] - last_centroids[id][1])**2)**0.5  # Euclidean distance between last and current centroids
            timer = (ender[id] - starter[id])
            speed = (int(distance)/timer)
            cvzone.putTextRect(img, f'{id} {speed:.2f}m/s', (x1,y1), scale=1, thickness=1, colorR=(0,0,255))


    cv2.imshow('Image', img)
    if cv2.waitKey(1) == ord('q'):
        break
