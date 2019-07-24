import numpy as np
'''
NOTE: If importing the cv2 library gives you issues; install opencv-python instead
'''
import cv2 as cv
import random
import math
global paused,startX,startY,rectPts,selectedKeypoints,keypoints
startX = 0
startY = 0
rectPts = {}
paused = False
keypoints = None #Keypoints are blobs/points that the OpenCV classifier detects
selectedKeypoints = False
keypointsLabeled = {}

'''
https://drive.google.com/file/d/1eOvLamj6X_1wbLHkNcgyFj2a7N2L-ee7/view?usp=sharing 
Download and move to the same directory as this python file; its the footage I used.
'''
videoFeed = cv.VideoCapture("LarvaFootage1.mp4")
params = cv.SimpleBlobDetector_Params() #Creates object in which parameters are stored

'''
The following parameters below filter the original video feed BEFORE any processing on our end is done to it.
I set the filters to be as general as possible since the larvae are literarlly small black dots, which is why the program detects a ton of random blobs.
'''


#https://answers.opencv.org/question/60374/simpleblobdetector-threshold/ refer to the answer of this post.
params.minThreshold = 0
params.maxThreshold = 500

#https://www.learnopencv.com/blob-detection-using-opencv-python-c/ Refer to this for the other parameter descriptions

# Filter by Area
params.filterByArea = False #Enable/Disable
params.minArea = 1500

# Filter by Circularity
params.filterByCircularity = False
params.minCircularity = 0.1

# Filter by Convexity
params.filterByConvexity = False
params.minConvexity = 0.87

# Filter by Inertia
params.filterByInertia = False
params.minInertiaRatio = 0.01

detector = cv.SimpleBlobDetector_create(params) #The actual detector

if (videoFeed.isOpened() == False):
    print("Error opening video stream or file")


'''
@brief Returns a random RGB color value
@:return Random color value as a tuple in RGB format
'''
def random_color():
    rgbl=[255,0,0]
    random.shuffle(rgbl)
    return tuple(rgbl)

'''
@brief Callback function; tied to ANY mouse event. Region of Interest (ROI) box. Allows user to drag a box on the video feed.
@:param event Returns the current event triggered by the mouse.
@:param x X Position of mouse cursor
@:param y Y Position of mouse cursor
'''
def captureClick(event,x,y,flags,params):
    global paused,rectPts,startX,startY,selectedKeypoints
    if not selectedKeypoints:
        if (event == cv.EVENT_LBUTTONDOWN):
            if (not paused):
                startX, startY = x, y
            paused = True
            rectPts = [(startX,startY)]
        elif (event == cv.EVENT_LBUTTONUP):
            paused = False
            selectedKeypoints = True
            rectPts = [(startX, startY)]
            rectPts.append((x, y))
        if (event == cv.EVENT_MOUSEMOVE and paused):
            rectPts = [(startX, startY)]
            rectPts.append((x, y))

'''
@brief Finds any Keypoints located within the ROI define above
@:return points An array of the Keypoints located within the ROI
'''
def getKeypointsWithinBox():
    points = []
    if keypoints != None:
        for point in keypoints:
            #bottom right quadrant
            if (rectPts[1][0] > rectPts[0][0] and rectPts[1][1] > rectPts[0][1]):
                if (point.pt[0] > rectPts[0][0] and point.pt[0] < rectPts[1][0] and point.pt[1] > rectPts[0][1] and point.pt[1] < rectPts[1][1]):
                   points.append(point)
            #upper right quadrant
            if (rectPts[1][0] > rectPts[0][0] and rectPts[1][1] < rectPts[0][1]):
                if (point.pt[0] > rectPts[0][0] and point.pt[0] < rectPts[1][0] and point.pt[1] < rectPts[0][1] and point.pt[1] > rectPts[1][1]):
                   points.append(point)
            #bottom left quadrant
            if (rectPts[1][0] < rectPts[0][0] and rectPts[1][1] > rectPts[0][1]):
                if (point.pt[0] < rectPts[0][0] and point.pt[0] > rectPts[1][0] and point.pt[1] > rectPts[0][1] and point.pt[1] < rectPts[1][1]):
                   points.append(point)
            #upper left quadrant
            if (rectPts[1][0] < rectPts[0][0] and rectPts[1][1] < rectPts[0][1]):
                if (point.pt[0] < rectPts[0][0] and point.pt[0] > rectPts[1][0] and point.pt[1] < rectPts[0][1] and point.pt[1] > rectPts[1][1]):
                   points.append(point)
    return points

'''
@brief Takes in a keypoint 'prevPoint' and an array of points 'points' to find the closest point while accounting for the fact that keypoints may not be the same in every frame.
i.e the keypoint youre tracking in one frame may not be registered in the next frame.
@:param prevPoint The previous position of the keypoint that is being tracked.
@:param points Array containing all the points to compare against.
@:param thresHold Max distance, in pixels, a keypoint can be before it is ignored
@:return closestPoint The keypoint closest to the previous point.
'''
def getMostAccurateKeypoint(prevPoint,points, thresHold):
    closestPoint = points[0]
    for point in points:
        skip = False
        if abs(point.pt[0] - prevPoint.pt[0]) > thresHold or abs(point.pt[1] - prevPoint.pt[1]) > thresHold:
            skip = True
        if not skip:
            if abs(point.pt[0] - prevPoint.pt[0]) < abs(closestPoint.pt[0] - prevPoint.pt[0]) and abs(point.pt[1] - prevPoint.pt[1]) < abs(closestPoint.pt[1] - prevPoint.pt[1]):
                closestPoint = point
    if abs(closestPoint.pt[0] - prevPoint.pt[0]) > thresHold or abs(closestPoint.pt[1] - prevPoint.pt[1]) > thresHold:
        return prevPoint
    return closestPoint


cv.namedWindow('Keypoints') #Creates a window called Keypoints
cv.setMouseCallback('Keypoints', captureClick) #Calls captureClick whenever any mouse event is fired within the Keypoints window

while (videoFeed.isOpened()):
    global frame
    if (not paused):
        ret, frame = videoFeed.read() #Captures the next frame in the video. Everytime this is called, the video will progress one frame.
        if not selectedKeypoints: #If you havent selected any keypoints to track yet; then display footage normally
            if ret == True: #If theres a frame
                gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY) #Grayscales the frame
                keypoints = detector.detect(gray) #Runs the detector on the grayscale frame
                frame = cv.drawKeypoints(frame, keypoints, gray,cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) #Draws circles on the keypoints
                cv.imshow('Keypoints', frame) #Shows the frame
            else:
                break
        elif len(keypointsLabeled) > 0 and len(rectPts) > 1: #If youve selected keypoints and theres been at least 1 keypoint detected, begin tracking
            '''
            Ideally, you'd want to run the detector on only the small portion of the screen selected, this would reduce computation time and lag.
            However, you'd have to track each keypoint and create a new box around each keypoint on which to run the detector. I couldnt get it working when I tried
            that, so I defaulted to using the entire frame.
            '''
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) #Grayscales frame
            keypoints = detector.detect(gray) #Detects keypoints in frame
            for i in keypointsLabeled: #Goes through each keypoint detected within the ROI
                newPt = getMostAccurateKeypoint(keypointsLabeled[i],keypoints,10) #The best estimate of the closest point to the current keypoint
                keypointsLabeled[i] = newPt #Sets the current point to the new point
                cv.circle(frame,(int(newPt.pt[0]),int(newPt.pt[1])),2,(0,0,255),2) #Draws a nice blue circle on the new point
                cv.putText(frame,str(i),(int(newPt.pt[0]),int(newPt.pt[1])),cv.FONT_HERSHEY_COMPLEX_SMALL,1,(255,0,0)) #Adds a label to the point
                cv.drawKeypoints(frame, keypoints, frame, cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) #Draws all the keypoints
                cv.imshow('Keypoints',frame) #Shows the frame

    elif paused: #If paused;
        #This is when the ROI is drawn and the mouse is moving
        if (len(rectPts) > 1):  #If theres any ROI drawn
            clone = frame.copy() #Clone the frame so we can draw on it
            cv.rectangle(clone, rectPts[0], rectPts[1], (0, 255, 0), 2) #Draws ROI rectangle with color green and thickness 2
            keypointsWithin = getKeypointsWithinBox() #Gets all keypoints within the ROI
            counter = 1 #Used for making labels
            for point in keypointsWithin:
                cv.circle(clone,(int(point.pt[0]),int(point.pt[1])),2,(0,0,255),2) #Draws a blue circle on point with thickness 2
                cv.putText(clone,str(counter),(int(point.pt[0]),int(point.pt[1])),cv.FONT_HERSHEY_COMPLEX_SMALL,1,(255,0,0)) #Adds text to the keypoint
                keypointsLabeled[counter] = point #Puts the labeled keypoint in the array
                counter += 1
            cv.imshow('Keypoints',clone) #Shows frame
    # Press Q on keyboard to  exit
    if cv.waitKey(25) & 0xFF == ord('q'):
        break
videoFeed.release()
cv.destroyAllWindows()