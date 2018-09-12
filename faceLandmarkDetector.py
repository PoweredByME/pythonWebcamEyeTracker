import dlib;
import cv2;
import imutils;
import utils;
from imutils import face_utils;

DLIB_FACE_LANDMARK_DETECTOR_PATH = "shape_predictor_68_face_landmarks.dat";

class faceLandmarkDetector(object):
    def __init__(self):
        self._landmarkPredictorPath = DLIB_FACE_LANDMARK_DETECTOR_PATH;
        self._landmarkPredictor = dlib.shape_predictor(self._landmarkPredictorPath);
        self._faceDetector = dlib.get_frontal_face_detector();

    def setFrame(self, frame, frameResizedWidth = 400):
        self._frame = frame;
        self._originalFrame = frame;
        l = len(self._frame.shape);
        if l == 3:
            # image is RGB. Convert to grey scale
            self._frame = cv2.cvtColor(self._frame, cv2.COLOR_BGR2GRAY);
        elif l > 3:
            raise Exception("A grayscale or RGB image is required.");
        self._resizeRatio, self._resizedFrame = utils.frameResize(self._frame, new_width = frameResizedWidth);

    def _detectFaces(self):
        return self._faceDetector(self._resizedFrame, 0);

    def _detectLandmarks(self):
        faces = self._detectFaces();

        self._faceLandmarksList = []; # an array that contains tuples containing a face and the respective facial landmarks.

        for rect in faces:
            landmarks = self._landmarkPredictor(self._resizedFrame, rect);
            np_landmarks = face_utils.shape_to_np(landmarks);

            landmarksList = [];
            for (x,y) in np_landmarks:
                rx = int (x / self._resizeRatio);
                ry = int (y / self._resizeRatio);
                landmarksList.append((rx,ry));
            
            # dlib rectangle to cv bounding box
            rectx = int(rect.left() / self._resizeRatio);
            recty = int(rect.top() / self._resizeRatio);
            rectw = int((rect.right() - rect.left()) / self._resizeRatio);
            recth = int((rect.bottom() - rect.top()) / self._resizeRatio);
            rect = (rectx, recty, rectw, recth);
            
            self._faceLandmarksList.append((rect, landmarksList))
                

    def detect(self):
        self._detectLandmarks();
        return self._faceLandmarksList;