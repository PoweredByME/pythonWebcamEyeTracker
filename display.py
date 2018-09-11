import cv2;

class Display(object):
    def __init__(self):
        pass;

    def showFrame(self, frame, windowName = "frame"):
        cv2.namedWindow(windowName, cv2.WINDOW_NORMAL);
        cv2.imshow(windowName, frame);

    def end(self):
        cv2.destroyAllWindows();