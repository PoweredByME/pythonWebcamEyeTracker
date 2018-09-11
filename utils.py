import cv2;

def CV_isButtonPressed(key = "q"):
    return cv2.waitKey(1) & 0xFF == ord(key);

def CV_waitForAnyKeyToBePressed():
    cv2.waitKey(0);