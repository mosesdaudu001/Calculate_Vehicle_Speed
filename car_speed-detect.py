from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
import os

class ObjectDetection():

    def __init__(self, capture, result):
        self.capture = capture
        self.result = result
        self.model = self.load_model()
        self.CLASS_NAMES_DICT = self.model.model.names

    def load_model(self):
        model = YOLO("yolo_weights/yolov8l.pt") 
        model.fuse()

        return model
    
    def predict(self, img):
        results = self.model(img, stream=True)
        return results
    
    def plot_boxes(self, results, img, detections):

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1,y1,x2,y2 = box.xyxy[0]
                x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
                w,h = x2-x1, y2-y1

                # Classname
                cls = int(box.cls[0])
                currentClass = self.CLASS_NAMES_DICT[cls]

                # Confodence score
                conf = math.ceil(box.conf[0]*100)/100

                if conf > 0.5:
                    currentArray = np.array([x1,y1,x2,y2,conf])
                    detections = np.vstack((detections, currentArray))
                    
        return detections, img
   
    def track_detect(self, detections, img, tracker, limitsUp, limitsDown, start_time, end_time, last_centroids, current_centroids, starter, ender):
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

        return img

    def __call__(self):

        cap = cv2.VideoCapture(self.capture)
        assert cap.isOpened()

        result_path = os.path.join(self.result, 'results.avi')

        codec = cv2.VideoWriter_fourcc(*'XVID')
        vid_fps =int(cap.get(cv2.CAP_PROP_FPS))
        vid_width,vid_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(result_path, codec, vid_fps, (vid_width, vid_height))

        tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

        limitsUp = [334, 121, 544, 111]
        limitsDown = [286, 247, 647, 235]
        start_time = 0
        end_time = 0
        last_centroids = {}
        current_centroids = {}
        starter = {}
        ender = {}

        if not os.path.exists(self.result):
            os.makedirs(self.result)
            print("Result folder created successfully")
        else:
            print("Result folder already exist")

        while True:

            _, img = cap.read()
            assert _
            
            detections = np.empty((0,5))
            results = self.predict(img)
            detections, frames = self.plot_boxes(results, img, detections)
            detect_frame = self.track_detect(detections, frames, tracker, limitsUp, limitsDown, start_time, end_time, last_centroids, current_centroids, starter, ender)

            out.write(detect_frame)
            cv2.imshow('Image', detect_frame)
            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        
        
    
detector = ObjectDetection(capture="Videos/traffic.mp4", result='result')
detector()

