import cv2;

def CV_isButtonPressed(key = "q"):
    return cv2.waitKey(1) & 0xFF == ord(key);

def CV_waitForAnyKeyToBePressed():
    cv2.waitKey(0);

def frameResize(frame, new_width):
    h,w,c = 0,0,0;
    if len(frame.shape) > 2:
        h, w, c = frame.shape;
    else:
        h, w = frame.shape;
    ratio = float(new_width) / float(w);
    dim = (int(new_width), int(float(h) * ratio));
    frame = cv2.resize(frame, dim, interpolation= cv2.INTER_LINEAR);    
    return (ratio, frame);