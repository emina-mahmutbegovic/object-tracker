# import the necessary packages
from collections import OrderedDict
from sklearn.utils.linear_assignment_ import linear_assignment
import numpy as np


class CentroidTracker():
    def __init__(self, 
                 initialFrameTimestamp,
                 maxDisappeared=10, 
                 minHits=1, 
                 minSumThreshold = 10, 
                 iouThrd = 0.3,
                 filterMultiple = 40,
                 Q = 0.1):
        # initialize the next unique object ID along with two ordered
        # dictionaries used to keep track of mapping a given object
        # ID to its centroid and number of consecutive frames it has
        # been marked as "disappeared", respectively
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        
        # dictionary to keep track of new detections and number of hits with
        # the tracked ones. If number of hits reaches the predefined min number
        # register that detection as a new object
        self.hits = OrderedDict()
        
        # array for frame timestamps for calculating parameter in
        # state transition matrix F of the Kalman filter
        self.frameTimestamps = []
        self.frameTimestamps.append(initialFrameTimestamp)

        # store the number of maximum consecutive frames a given
        # object is allowed to be marked as "disappeared" until we
        # need to deregister the object from tracking
        self.maxDisappeared = maxDisappeared
        
        # store the number of minimum consecutive hits (overlapping)
        # to consider that the two objects that are overlapping are
        # actually the same
        self.minHits = minHits
        
        # min sum threshold for the comparison of the two objects
        self.minSumThreshold = minSumThreshold
        
        # store the number of minimum iou threshold, to consider
        # that two objects are overlapping
        self.iouThrd = iouThrd
        
        # store maximum difference value between detections, to 
        # determine if it is for the same object
        self.filterMultiple= filterMultiple
        
         # state vector
        self.x = OrderedDict()
        
        # covariance matrix
        self.P = OrderedDict()
        
        # state transition matrix
        self.F = np.identity(8, dtype="float32")
        
        # noise value
        self.Q = Q
        
        # measurement matrix
        self.H = np.zeros((4, 8), dtype="int")
        for i in range (0, 4):
            self.H[i, i] = 1
        
        # measurement noise
        self.R = np.identity(4, dtype="int")
        self.R[2, 2] = 20
        self.R[3, 3] = 20
        
    # ispraviti detection
    def register(self, detection):
        # when registering an object we use the next available object
        # ID to store the centroid
        self.objects[self.nextObjectID] = detection
        # initalize state space variable
        self.x[self.nextObjectID] = np.concatenate((self.setStateArray(detection), 
                                                    np.zeros((4, ), dtype="int32")), 
                                                    axis=None)
        # initialize covariance matrix
        self.P[self.nextObjectID] = np.identity(8, dtype="int")*20
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1
        
    def deregister(self, objectID):
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        del self.objects[objectID]
        del self.x[objectID]
        del self.P[objectID]
        del self.disappeared[objectID]
    
    def boxes2iou(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        # return the intersection over union value
        return iou
    
    # another implementation of iou calculation function
    def iou2(self, boxA, boxB):
        
        w_intsec = np.maximum (0, (np.minimum(boxA[2], boxB[2]) - np.maximum(boxA[0], boxB[0])))
        h_intsec = np.maximum (0, (np.minimum(boxA[3], boxB[3]) - np.maximum(boxA[1], boxB[1])))
        s_intsec = w_intsec * h_intsec
        s_a = (boxA[2] - boxA[0])*(boxA[3] - boxA[1])
        s_b = (boxB[2] - boxB[0])*(boxB[3] - boxB[1])
  
        return float(s_intsec)/(s_a + s_b -s_intsec)
    
    # find a key for particular tracking in the list
    # if key wasn't found return -1
    def findObjectKey(self, tracking):
        
        for key, value in self.objects.items():
            sums = self.compareCoordinateArrays(value, tracking)
            # if they are exactly the same break, else continue
            if(sums == 0): return key
        
        # if the key isn't found return -1
        return -1
        
    
    # function for comparison of the coordinate arrays
    # coordinates of the bounding boxes of the same object can differ
    # in a few pixels, but it is stil the same object
    def compareCoordinateArrays(self, obj, tracking):
               return (abs(obj[0] - tracking[0]) + 
                       abs(obj[1] - tracking[1]) + 
                       abs(obj[2] - tracking[2]) +
                       abs(obj[3] - tracking[3]))
    
    # method for calculating med array value
    # help method for filtering multiple detections
    def calculateMediumArray(self, arrayA, arrayB):
            returnValue = []
            # if the lenghts of inputs are different, an error
            # occured
            if len(arrayA) != len(arrayB):
                return -1
            else:
                for i in range(len(arrayA)):
                    returnValue.append(int((arrayA[i] + arrayB[i])/2))
            return returnValue
    
    # function for filtering out multiple detections for the same
    # object
    # if the difference between two dectections is very small,
    # there is a possibility that it is the same object
    def filterMultipleDetections(self, detections):
        # an array for storing the difference sum
        sums = OrderedDict()
        # side array for choosing between detections
        medArray = []
        # dict format of the detections
        dictDetections = OrderedDict()
        dictDetections = self.listToDict(detections)
        # an array for storing indexes of the close arrays
        for d, det in enumerate(detections):
            if (d != len(detections) - 1):
                nextIndex = d + 1
                for dNext, detNext in enumerate(detections[(d+1):]):
                    difference = self.compareCoordinateArrays(det, detNext)
                    sums[(d, nextIndex)] = difference
                    nextIndex += 1
        # loop through the sums to find if some sum is lower than 
        for key, value in sums.items():
            if value <= self.filterMultiple:
                medArray = self.calculateMediumArray(detections[key[0]], 
                                                     detections[key[1]])
                if(medArray != -1):
                    abs1 = self.compareCoordinateArrays(detections[key[0]],
                                                        medArray)
                    abs2 = self.compareCoordinateArrays(detections[key[1]],
                                                        medArray)
                    if(abs1 < abs2):
                        try:
                            del dictDetections[key[1]]
                        except:
                            print("ERROR: Key not found!")
                    elif (abs1 == abs2):
                        try:
                            del dictDetections[key[0]]
                        except:
                            print("ERROR: Key not found!")
                    else:
                        try:
                            del dictDetections[key[0]]
                        except:
                            print("ERROR: Key not found!")
                else:
                    print("ERROR: Wrong dimensions of the input arrays")
        
        returnValue = []
        for detections in dictDetections.values():
            returnValue.append(detections)
        
        return returnValue
    
    # simple list do dict converter
    def listToDict(self, inputList): 
        outputDict = OrderedDict()
        for i in range(len(inputList)):
            outputDict[i] = inputList[i]
        return outputDict
    
    def setFMatrix(self, dt):
        innerCount = 4
        for i in range(0,4):
            self.F[i, innerCount] = dt
            innerCount += 1
            if innerCount > 7: break
        
    def setStateArray(self, detection):
        # coordinates are arranged in a following way:
        # upper left corner - (detection[0], detection[1])
        # lower right corner - (detection[2], detection[3])
        
        width = detection[2] - detection[0]
        height = detection[3] - detection[1] 
        cx = detection[0] + int(width / 2) 
        cy = detection[3] - int(height / 2)
        
        return [cx, cy, width, height]
        
        
        
    
    def predict(self, x, key):
        # calculate dt
        dt = (self.frameTimestamps[-1] - self.frameTimestamps[-2])/1000
        # set F matrix
        self.setFMatrix(dt)
        # predict x and P
        xPredicted = self.F.dot(x)
        pPredicted = (self.F.dot(self.P[key])).dot(self.F.transpose()) + self.Q
        
        # return predicted values
        return [xPredicted, pPredicted]
    
    def kalmanUpdate(self, predicted, z):
        xPredicted = predicted[0]
        pPredicted = predicted[1]
        
        y = z - self.H.dot(xPredicted)
        S = (self.H.dot(pPredicted)).dot(self.H.transpose()) + self.R
        K = (pPredicted.dot(self.H.transpose())).dot(np.linalg.inv(S))
        
        xCorrected = xPredicted + K.dot(y)
        pCorrected = (np.identity(8) - K.dot(self.H)).dot(pPredicted)
        
        return [xCorrected, pCorrected]
    
    def kalmanFilter(self, x, z, key):
        predicted = self.predict(x, key)
        corrected = self.kalmanUpdate(predicted, z)
        
        return corrected

    def update(self, detections, frameTimestamp):
        
        # collect frame timestamps for calculation dt between 
        # current and previous frame
        self.frameTimestamps.append(frameTimestamp)
        
        # check to see if the list of input bounding box rectangles
        # is empty
        if len(detections) == 0:
            # loop over any existing tracked objects and mark them
            # as disappeared
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1

                # if we have reached a maximum number of consecutive
                # frames where a given object has been marked as
                # missing, deregister it
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            # return early as there are no centroids or tracking info
            # to update
            return self.objects
        
        # filter out multiple detections
        detections = self.filterMultipleDetections(detections)
        
        
        # initialize an array of input centroids for the current frame
        inputCentroids = np.zeros((len(detections), 2), dtype="int")
        # initialize an array of input detections for the current frame
        inputDetections = np.zeros((len(detections), 4), dtype="int")
        #initialize an array of IOU between current and previous detections
        IOU_mat= np.zeros((len(detections), len(self.objects.values())), dtype=np.float32)

        count = 0
        
        # loop over the bounding box rectangles
        for (i, (startX, startY, endX, endY)) in enumerate(detections):
            inputCentroids[i] = (startX, startY)
            inputDetections[i] = (startX, startY, endX, endY)
            count = count + 1
        
        # if we are currently not tracking any objects take the input
        # centroids and register each of them
        if len(self.objects) == 0:
            for i in range(0, len(inputDetections)):
                self.register(inputDetections[i])
        # otherwise, are are currently tracking objects so we need to
        # try to match the input centroids to existing object
        # centroids
        else:
            # grab coordinates of tracked objects
            trackings = list(self.objects.values())
            # grab centroids of tracked objects
            objectCentroids = np.zeros((len(trackings), 2), dtype="int")
            for i, coordinates in enumerate(trackings):
                objectCentroids[i] = (coordinates[0], coordinates[1])
                
            #calculating matrix of IOUs between detected and tracked objects
            for d, det in enumerate(detections):
                for o, obj in enumerate(trackings):
                    IOU_mat[d,o] = self.boxes2iou(det, obj)
         
            # applying Munkres algorithm to do the tracking
            matched_indexes_linear = linear_assignment(-IOU_mat)
            
            #i nitializing an array for the unmatched trackers and detections
            unmatched_trackers, unmatched_detections = [], []
            
            # initializing an array for debugging
            matches = []
            
            #calculating matrix of IOUs between detected and tracked objects
            for d, det in enumerate(detections):
                if(d not in matched_indexes_linear[:,0]):
                    unmatched_detections.append(d)
            for o, obj in enumerate(trackings):
                if(o not in matched_indexes_linear[:, 1]):
                    unmatched_trackers.append(o)
           
            # iterate through the matchings
            for m in matched_indexes_linear:
                # if matching is below the predefined iou threshold reject it
                if(IOU_mat[m[0], m[1]] < self.iouThrd):
                    unmatched_detections.append(m[0])
                    unmatched_trackers.append(m[1])
                else:
                    matches.append(m)
                    # associate tracking and the detection for the found match
                    # first we need to obtain object key of the tracking
                    key = self.findObjectKey(trackings[m[1]])
                    # if the key is found update the trackings with the
                    # associated detection
                    if(key != -1):
                        self.objects[key] = detections[m[0]]
                        # get new measurement
                        measurement = self.setStateArray(detections[m[0]])
                        # implement Kalman Filter
                        [xCorrected, pCorrected] = self.kalmanFilter(self.x[key], 
                                                                    measurement, key)
                        # update state and covariance matrix
                        self.x[key] = xCorrected
                        self.P[key] = pCorrected
                        
            # iterate through unmatched detections to register new objects
            for det in unmatched_detections:
                # this list is a key to the hits dictionary and must be 
                # converted to tuple because the key must be immutable
                detection = detections[det]
                self.register(detection)
                # check if there are no entries in the hits dictionary
                # if so, initialize number of hits
                #if(self.hits.get(tuple(detection), -1) == -1):
                 #   self.hits[tuple(detection)] = 1
                #else:
                 #   self.hits[tuple(detection)] += 1
                #if(self.hits[tuple(detection)] > self.minHits):
                 #   self.register(detections[det])
                  #  self.hits.pop(tuple(detection))
            
            # iterate through unmatched trackings to deregister disappeared
            # objects
            for tracking in unmatched_trackers:
                key = self.findObjectKey(trackings[tracking])
                # if the object exists in the trackings
                if(key != -1):
                    # increment a number of consecutive disappearings
                    self.disappeared[key]+=1;
                    # if number of consecutive disappearings is bigger than
                    # predefined number of max consecutive disappearings
                    # deregister
                    if(self.disappeared[key] > self.maxDisappeared):
                        self.deregister(key)
            
        return self.x, self.objects