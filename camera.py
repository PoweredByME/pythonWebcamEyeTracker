import cv2;


class Camera(object):
    def __init__(self):
        self._cv_VideoSource = None;       # Uasually a Camera object
        self._cv_imageSource = None;       # Uasually a string containing a path to the image

    def setVideoSource(self, source):
        self._cv_VideoSource = cv2.VideoCapture(source);
        
    def isVideoSourceAvaiable(self):
        return self._cv_VideoSource.isOpened();

    def getVideoFrame(self):
        if self._cv_VideoSource is None:
            raise Exception("The video source is not set");
        self._videoSourceAvailablityException();
        ret, frame = self._cv_VideoSource.read();
        return ret, frame;

    def setImageSource(self, source):
        self._cv_imageSource = source;
    
    def getImage(self):
        return cv2.imread(self._cv_imageSource);

    def _videoSourceAvailablityException(self):
        # Raise exception if the video source is not available.
        if not self.isVideoSourceAvaiable():
            raise Exception("The video source is not available");

    def end(self):
        self._cv_VideoSource.release();
