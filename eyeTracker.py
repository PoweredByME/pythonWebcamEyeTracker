import dlib;
import cv2;
from imutils import face_utils;
import numpy as np;
import utils;
import cythonUtils as cutils;

'''
    This code is a implementation of the method proposed in the paper by Fabian Timm
    http://www.inb.uni-luebeck.de/publikationen/pdfs/TiBa11b.pdf

    The method used for calculating the point of the pupil is inspired from the
    method by Tristan Hume.
    https://github.com/trishume/eyeLike
'''

class eyeTracker(object):
    def __init__(self):
        pass;
    
    def setParameters(self, frame, face, landmarksList):
        self._frame = frame;
        self._face = face;
        self._landmarksList = landmarksList;


    def _getEyeLandmarks(self):
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        leftEyeLandmarks = self._landmarksList[lStart : lEnd];
        rightEyeLandmarks = self._landmarksList[rStart: rEnd];
        return leftEyeLandmarks, rightEyeLandmarks;
    
    def Process(self):
        frame = cv2.cvtColor(self._frame, cv2.COLOR_BGR2GRAY);


        left, right = self._getEyeLandmarks();
        npl = np.asarray(left);
        npr = np.asarray(right);        
        leftEyeHull = cv2.convexHull(npl);
        rightEyeHull = cv2.convexHull(npr);
        maskLeftEye = np.ones_like(frame) * 255;
        maskRightEye = np.ones_like(frame) * 255;
        cv2.drawContours(maskLeftEye, [leftEyeHull], -1, 0, -1);
        cv2.drawContours(maskRightEye, [rightEyeHull], -1, 0, -1);
        outLeft = np.ones_like(frame) * 255;
        outLeft[maskLeftEye == 0] = frame[maskLeftEye == 0];
        #outLeft[maskLeftEye == 0] = outLeft[maskLeftEye == 0];
        
        outRight = np.ones_like(frame) * 255;
        outRight[maskRightEye == 0] = frame[maskRightEye == 0];
        #outRight[maskRightEye == 0] = outRight[maskRightEye == 0];

        #croping the left eyes
        (x, y) = np.where(maskLeftEye == 0);
        (topx, topy) = (np.min(x), np.min(y));
        (bottomx, bottomy) = (np.max(x), np.max(y));
        outLeft = frame[topx:bottomx + 1, topy:bottomy];
        ltopx = topx;
        ltopy = topy;

        #croping the right eyes
        (x, y) = np.where(maskRightEye == 0);
        (topx, topy) = (np.min(x), np.min(y));
        (bottomx, bottomy) = (np.max(x), np.max(y));
        outRight = frame[topx:bottomx + 1, topy:bottomy];
        rtopx = topx;
        rtopy = topy;
        

        o = outLeft;
        olRatio, outLeft = utils.frameResize(outLeft, 30);
        orRatio, outRight = utils.frameResize(outRight, 30);
        
        (clx,cly) = self._trackEyePupil(outLeft, olRatio);
        (crx,cry) = self._trackEyePupil(outRight, orRatio);


        leftCenter = (ltopy + clx, ltopx + cly);
        rightCenter = (rtopy + crx, rtopx + cry); 
        return (leftCenter, rightCenter);




    def _computeDynamicThreshold(self, frame):
        stdDevFactor = 25.0;
        (meanGradMagn, stdGradMagn) = cv2.meanStdDev(frame);
        h, w = frame.shape;
        stdDev = (stdGradMagn[0][0]) / ((h*w)**0.5);
        return stdDevFactor * stdDev + meanGradMagn[0][0];


    def _trackEyePupil(self, eye, resizeRatio):
        # preprocessing the eye image for algorithm
        temp = np.zeros(eye.shape);
        gx = cutils.calcGradientX(eye, eye.shape[0], eye.shape[1], temp);
        gx = np.asarray(gx);
        _eye = cv2.transpose(eye);
        temp = np.zeros(_eye.shape);
        gy = cutils.calcGradientX(_eye, _eye.shape[0], _eye.shape[1], temp);
        gy = cv2.transpose(np.asarray(gy));
        magn = gx ** 2 + gy ** 2;
        magn = magn ** 0.5;
        gradThresh = self._computeDynamicThreshold(magn);
        gx = gx / magn; #   normalization
        gy = gy / magn; #   normalization
        eye = cv2.GaussianBlur(eye, (5,5), 0);
        inv_eye = 255 - eye;    #   inverting image
        
        h, w = eye.shape;
        outSum = np.zeros(eye.shape);
        outSum = np.asarray(cutils.findEyeCenter(inv_eye, gx, gy, magn, h, w, outSum));
        outSum = outSum / (h * w);

        (minVal, maxLoc, minLoc, maxLoc) =  cv2.minMaxLoc(outSum);
        (x,y) = (maxLoc[0], maxLoc[1]);
        
        return (int(x/resizeRatio), int(y/resizeRatio));
