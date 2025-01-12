import cv2 as cv
import argparse
import sys
import numpy as np
import os
import dlib
from imutils.video import FPS

# Import libraries for tracking
from tracking.centroidtracker import CentroidTracker
from tracking.trackableobject import TrackableObject

# Set OpenCV OCL4DNN configuration path
os.environ["OPENCV_OCL4DNN_CONFIG_PATH"] = "C:/path_to_your_cache_directory"

# Configuration parameters
confThreshold = 0.5  
nmsThreshold = 0.4   
inpWidth = 416       
inpHeight = 416      
skip_frames = 30     

parser = argparse.ArgumentParser(description='Object Detection and Tracking using YOLO in OpenCV')
parser.add_argument('--image', help='Path to image file.')
parser.add_argument('--video', help='Path to video file.')
parser.add_argument("-s", "--skip_frames", type=int, default=30,
                    help="# of skip frames between detections")
args = parser.parse_args()

# Load class labels
classesFile = "model/coco.names"
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Load YOLO model
modelConfiguration = "model/yolov3.cfg"
modelWeights = "model/yolov3.weights"
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_OPENCL)

def getOutputsNames(net):
    layersNames = net.getLayerNames()
    output_layers = net.getUnconnectedOutLayers()
    return [layersNames[i - 1] for i in output_layers]

def MarkPeople(objects, total):
    count = 0
    for (objectID, centroid) in objects.items():
        to = trackableObjects.get(objectID, None)

        if to is None:
            to = TrackableObject(objectID, centroid)
        else:
            to.centroids.append(centroid)
            if not to.counted:
                total += 1
                to.counted = True

        trackableObjects[objectID] = to
        text = "ID {}".format(objectID)
        cv.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
        count += 1

    return count, total

def Fill_tracker_list(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIds, confidences, boxes = [], [], []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold and classes and classes[classId] == "person":
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    trackers = []  # Initialize trackers list
    for i in indices.flatten():
        box = boxes[i]
        left, top, width, height = box
        tracker = dlib.correlation_tracker()
        rect = dlib.rectangle(left, top, left + width, top + height)
        tracker.start_track(frame, rect)  # Change to frame
        trackers.append(tracker)

    return trackers

# Setup window for displaying results
winName = 'Deep Learning Object Detection in OpenCV using YOLO_v3'
cv.namedWindow(winName, cv.WINDOW_NORMAL)

# Set output file
outputFile = "yolo_out_py.avi"
if args.image:
    if not os.path.isfile(args.image):
        print("Input image file ", args.image, " doesn't exist")
    sys.exit(1)
    cap = cv.VideoCapture(args.image)
    outputFile = args.image[:-4] + '_yolo_out_py.jpg'
elif args.video:
    if not os.path.isfile(args.video):
        print("Input video file ", args.video, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.video)
    outputFile = args.video[:-4] + '_yolo_out_py.avi'
else:
    cap = cv.VideoCapture(0)  # Use webcam if no image or video is provided

if not args.image:
    vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,
                                (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

# Initialize the centroid tracker and other variables
ct = CentroidTracker(maxDisappeared=60, maxDistance=50)
trackers = []
trackableObjects = {}
total = 0

fps = FPS().start()
totalFrames = 0

# Main loop for processing video frames
while True:
    hasFrame, frame = cap.read()
    if not hasFrame:
        print("Done processing !!!")
        print("Output file is stored as ", outputFile)
        cv.waitKey(3000)
        break

    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    status = "Waiting"
    rects = []

    # Perform detection every `skip_frames` frames
    if totalFrames % args.skip_frames == 0:
        status = "Detecting"
        blob = cv.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
        net.setInput(blob)
        outs = net.forward(getOutputsNames(net))
        trackers = Fill_tracker_list(frame, outs)
    else:
        for tracker in trackers:
            status = "Tracking"
            tracker.update(rgb)
            pos = tracker.get_position()
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())
            rects.append((startX, startY, endX, endY))

    objects = ct.update(rects)
    count, total = MarkPeople(objects, total)

    # Prepare information to display on the frame
    info = [
        ("Total up till now: ", total),
        ("In Frame: ", count),
        ("Status: ", status),
    ]

    (H, W) = frame.shape[:2]
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv.putText(frame, text, (10, H - ((i * 20) + 20)),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Save output if it's an image
    if args.image:
        cv.imwrite(outputFile, frame.astype(np.uint8))

    # Display the output frame
    cv.imshow(winName, cv.resize(frame, (1200, 900)))
    key = cv.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    totalFrames += 1
    fps.update()

# Stop the FPS counter and print the results
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# Clean up
cap.release()
if not args.image:
    vid_writer.release()
cv.destroyAllWindows()