import numpy as np
import cv2 as cv
import random
import math
global paused,startX,startY,rectPts,selectedKeypoints,keypoints
startX = 0
startY = 0
rectPts = {}
paused = False
keypoints = None
selectedKeypoints = False
keypointsLabeled = {}
videoFeed = cv.VideoCapture("LarvaFootage1.mp4")
params = cv.SimpleBlobDetector_Params()


params.minThreshold = 0
params.maxThreshold = 500

# Filter by Area.
params.filterByArea = False
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

detector = cv.SimpleBlobDetector_create(params)

if (videoFeed.isOpened() == False):
    print("Error opening video stream or file")

def random_color():
    rgbl=[255,0,0]
    random.shuffle(rgbl)
    return tuple(rgbl)


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

def getMostAccurateKeypoint(prevPoint,points):
    closestPoint = points[0]
    for point in points:
        skip = False
        if abs(point.pt[0] - prevPoint.pt[0]) > 10 or abs(point.pt[1] - prevPoint.pt[1]) > 10:
            skip = True
        if not skip:
            if abs(point.pt[0] - prevPoint.pt[0]) < abs(closestPoint.pt[0] - prevPoint.pt[0]) and abs(point.pt[1] - prevPoint.pt[1]) < abs(closestPoint.pt[1] - prevPoint.pt[1]):
                closestPoint = point
    if abs(closestPoint.pt[0] - prevPoint.pt[0]) > 10 or abs(closestPoint.pt[1] - prevPoint.pt[1]) > 10:
        return prevPoint
    return closestPoint


cv.namedWindow('Keypoints')
cv.setMouseCallback('Keypoints', captureClick)

while (videoFeed.isOpened()):
    global frame
    if (not paused):
        # cv.waitKey(25) & 0xFF == ord('d'):
        ret, frame = videoFeed.read()
        if not selectedKeypoints:
            if ret == True:
                gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
                keypoints = detector.detect(gray)
                frame = cv.drawKeypoints(frame, keypoints, gray,cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                cv.imshow('Keypoints', frame)
                #if len(rectPts) == 2:
                    #roi = frame[rectPts[0][1]:rectPts[1][1], rectPts[0][0]:rectPts[1][0]]
            else:
                break
        elif len(keypointsLabeled) > 0 and len(rectPts) > 1:
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            keypoints = detector.detect(gray)
            for i in keypointsLabeled:
                newPt = getMostAccurateKeypoint(keypointsLabeled[i],keypoints)
                keypointsLabeled[i] = newPt
                cv.circle(frame,(int(newPt.pt[0]),int(newPt.pt[1])),2,(0,0,255),2)
                cv.putText(frame,str(i),(int(newPt.pt[0]),int(newPt.pt[1])),cv.FONT_HERSHEY_COMPLEX_SMALL,1,(255,0,0))
                cv.drawKeypoints(frame, keypoints, frame, cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                cv.imshow('Keypoints',frame)

    elif paused:
        if (len(rectPts) > 1):
            clone = frame.copy()
            cv.rectangle(clone, rectPts[0], rectPts[1], (0, 255, 0), 2)
            keypointsWithin = getKeypointsWithinBox()
            counter = 1
            for point in keypointsWithin:
                cv.circle(clone,(int(point.pt[0]),int(point.pt[1])),2,(0,0,255),2)
                cv.putText(clone,str(counter),(int(point.pt[0]),int(point.pt[1])),cv.FONT_HERSHEY_COMPLEX_SMALL,1,(255,0,0))
                keypointsLabeled[counter] = point
                counter += 1
            cv.imshow('Keypoints',clone)
    # Press Q on keyboard to  exit
    if cv.waitKey(25) & 0xFF == ord('q'):
        break
videoFeed.release()
cv.destroyAllWindows()