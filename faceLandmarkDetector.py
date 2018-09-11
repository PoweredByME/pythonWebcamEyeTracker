import dlib;
import cv2;
import imutils;

DLIB_FACE_LANDMARK_DETECTOR_PATH = "shape_predictor_68_face_landmarks.dat.bz2";

class faceLandmarkDetector(object):
    def __init__(self):
        self._landmarkPredictorPath = DLIB_FACE_LANDMARK_DETECTOR_PATH;
        self._landmarkPredictor = dlib.shape_predictor(self._landmarkPredictorPath);
        self._faceDetector = dlib.get_frontal_face_detector();

    def setFrame(self, frame, frameResizedWidth = 400):
        self._frame = frame;
        l = len(self._frame.shape):
        if l == 3:
            # image is RGB. Convert to grey scale
            self._frame = cv2.cvtColor(self._frame, cv2.COLOR_BGR2GRAY);
        elif l > 3:
            raise Exception("A grayscale or RGB image is required.");
        self._frame = imutils.resize(self._frame, width = frameResizedWidth);