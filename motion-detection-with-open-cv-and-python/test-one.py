import numpy as np
import cv2

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('motiondetect.avi',fourcc, 60.0, (1920,1080), isColor = False)

cap = cv2.VideoCapture(0) 
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)
cap.set(cv2.CAP_PROP_FPS, 30)

foreground = cv2.createBackgroundSubtractorMOG2(detectShadows = False)

while(True):
 
    (ret, camframe) = cap.read()

    grayframe = cv2.cvtColor(camframe, cv2.COLOR_BGR2GRAY)

    smallframe = cv2.resize(grayframe, (80,60))    

    blurframe = cv2.medianBlur(smallframe, 3)

    motionframe = foreground.apply(blurframe)

    cv2.imshow('motionframe',motionframe)
    cv2.imshow('blurframe',blurframe)

    detect = (np.sum(motionframe))//255
    if detect > 16:
        print("moving object size = ", detect)
        out.write(grayframe)
    k = cv2.waitKey(1) & 0xff
    if k == ord('s'):
        cap.set(cv2.CAP_PROP_SETTINGS, 0)
    if k == ord(' ') or k == 27 or ret == False:
        break

cap.release()
out.release()
cv2.destroyAllWindows()