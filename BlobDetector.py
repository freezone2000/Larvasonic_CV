import numpy as np
import cv2 as cv
global paused,startX,startY,rectPts
startX = 0
startY = 0
rectPts = {}
paused = False
videoFeed = cv.VideoCapture("Larvafootage1.mp4")
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


def captureClick(event,x,y,flags,params):
    global paused
    global rectPts
    global startX,startY
    if (event == cv.EVENT_LBUTTONDOWN):
        if (not paused):
            startX, startY = x, y
        paused = True
        rectPts = [(startX,startY)]
    elif (event == cv.EVENT_LBUTTONUP):
        paused = False
        rectPts = [(startX, startY)]
        rectPts.append((x, y))
    if (event == cv.EVENT_MOUSEMOVE and paused):
        rectPts = [(startX, startY)]
        rectPts.append((x, y))


cv.namedWindow('Keypoints')
cv.setMouseCallback('Keypoints', captureClick)
while (videoFeed.isOpened()):
    global frame
    if (not paused):
        ret, frame = videoFeed.read()
        if ret == True:
            gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
            keypoints = detector.detect(gray)
            frame = cv.drawKeypoints(frame, keypoints, gray,cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv.imshow('Keypoints',frame)
            #if len(rectPts) == 2:
                #roi = frame[rectPts[0][1]:rectPts[1][1], rectPts[0][0]:rectPts[1][0]]
        else:
            break
    elif paused:
        if (len(rectPts) > 1):
            clone = frame.copy()
            cv.rectangle(clone, rectPts[0], rectPts[1], (0, 255, 0), 2)
            cv.imshow('Keypoints',clone)
    # Press Q on keyboard to  exit
    if cv.waitKey(25) & 0xFF == ord('q'):
        break
videoFeed.release()
cv.destroyAllWindows()