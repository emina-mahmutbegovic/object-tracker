# USAGE
# python yolo_video.py --input videos/input_video.mp4 --output output/out.avi --yolo yolo-coco

# import the necessary packages
from centroidtracker import CentroidTracker
import numpy as np
import argparse
import imutils
import time
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
    help="path to input video")
ap.add_argument("-o", "--output", required=True,
    help="path to output video")
ap.add_argument("-y", "--yolo", required=True,
    help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
    help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
    dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(args["input"])
output = 'car_test.mp4'
writer = None

# initialize our centroid tracker and frame dimensions
ct = CentroidTracker(vs.get(cv2.CAP_PROP_POS_MSEC))
(W, H) = (None, None)

# variable for storing frames
frames = []

# try to determine the total number of frames in the video file
try:
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
        else cv2.CAP_PROP_FRAME_COUNT
    total = int(vs.get(prop))
    print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
    print("[INFO] could not determine # of frames in video")
    print("[INFO] no approx. completion time can be provided")
    total = -1
    
# load test images
    

# loop over frames from the video file stream
# images = [plt.imread(file) for file in glob.glob('./test_images/*.jpg')]
# for i in range(len(images))[0:7]:
  #  frame = images[i]
while True:
    
    # read the next frame from the file
    (grabbed, frame) = vs.read()
    frameTimestamp = vs.get(cv2.CAP_PROP_POS_MSEC)
 
    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break

    # if the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]
        
    # construct a blob from the input frame and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes
    # and associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
        swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # initialize our lists of detected bounding boxes, confidences,
    # and class IDs, respectively
    boxes = []
    # array for tracker
    detections = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        counter = 0
        for detection in output:
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > args["confidence"]:
                # scale the bounding box coordinates back relative to
                # the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of
                # the bounding box followed by the boxes' width and
                # height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                

                # use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates,
                # confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                detections.append([x, y, int(x + width), int(y + height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

                text = "{}: {:.4f}".format(LABELS[classIDs[counter]],
                confidences[counter])
                cv2.putText(frame, text, (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                #update counter
                counter = counter + 1


    # apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
        args["threshold"])

    filteredObjects, objects = ct.update(detections, frameTimestamp)
    
    # loop over the filtered tracked objects
    counter = 0
    for (objectID, detection) in filteredObjects.items():
        text = "ID {}".format(objectID)
        # draw both the ID of the object and the centroid of the
        # object on the output frame
        cx = int(detection[0])
        cy = int(detection[1])
        width = int(detection[2])
        height = int(detection[3])
        
        x = int(cx - (width / 2))
        y = int(cy - (height / 2))
        
        cv2.putText(frame, text, (cx, cy),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
        
        # draw a bounding box rectangle and label on the frame
        #color = [int(c) for c in COLORS[classIDs[counter]]]
        cv2.rectangle(frame, (x, y), (x + width, y + height),
                (0, 255, 0), 2)
        #update counter
        counter = counter + 1
    
    # loop over the unfiltered tracked objects
    counter = 0
    for (objectID, detection) in objects.items():
        # draw both the ID of the object and the centroid of the
        # object on the output frame
        width = detection[2]
        height = detection[3] 
        x = detection[0]
        y = detection[1] 
        # draw a bounding box rectangle and label on the frame
        cv2.rectangle(frame, (x, y), (width, height),
                (0, 0, 255) , 2)
        #update counter
        counter = counter + 1
    
    if writer is None:
        # initialize our video writer
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(args["output"], fourcc, 30,
                (frame.shape[1], frame.shape[0]), True)
        # some information on processing single frame
            if total > 0:
                elap = (end - start)
                print("[INFO] single frame took {:.4f} seconds".format(elap))
                print("[INFO] estimated total time to finish: {:.4f}".format(
                    elap * total))
        
    
    # write the output frame to disk
    writer.write(frame)
        

          

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()
