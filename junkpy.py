import cv2;
import numpy as np;
from utils import cythonUtils as cutils;


'''
    This function takes the camera object,
    processes the frames, and return bounding
    boxes of the faces in the frame.
'''
class FaceDetector(object):
    def __init__(
                self, 
                cam, 
                cv2_face_detector_file = 'data/haarcascades/haarcascade_frontalface_default.xml',
                cv2_eye_detector_file = 'data/haarcascades/haarcascade_eye_tree_eyeglasses.xml'
                ):
        self.camera = cam;
        self.faceFound = False;
        self.gotFrame = False;
        self.frame = None;
        self.bgrFrame = None;
        self.boundingBoxes_faces = [];
        self.boundingBoxes_eyes = {};
        # opencv face cascade classifier
        self.faceCascade = cv2.CascadeClassifier(cv2_face_detector_file);
        self.eyeCascade = cv2.CascadeClassifier(cv2_eye_detector_file);
        print self.faceCascade.empty();


    def getBGRFrame(self):
        return self.bgrFrame;

    def getGrayFrame(self):
        return self.frame;

    def getRGBFrame(self):
        return cv2.cvtColor(self.bgrFrame, cv2.COLOR_BGR2RGB);

    def getBoundingBoxes(self):
        return (self.boundingBoxes_faces, self.boundingBoxes_eyes);

    def isFaceFound(self):
        return self.gotFrame and self.faceFound;

    def getCamera(self):
        return self.camera;

    def isFrameCaptured(self):
        return self.gotFrame;

    # get the frame and preprocess it.
    def getFrame(self):
        self.gotFrame, self.bgrFrame = self.camera.read();
        if self.gotFrame:
            self.frame = cv2.flip(self.frame, 1);
            self.frame = cv2.GaussianBlur(self.frame, (5,5), 0);
            self.frame = cv2.cvtColor(self.bgrFrame, cv2.COLOR_BGR2GRAY);


    def setFrame(self, frame):
        self.gotFrame = True;
        self.bgrFrame = frame;
        self.frame = cv2.cvtColor(self.bgrFrame, cv2.COLOR_BGR2GRAY);

    def getFace_fullCascade(self):
        if self.gotFrame:
            #faces = self.faceCascade.detectMultiScale(self.frame);  
            faces = self.faceCascade.detectMultiScale(self.frame,1.1, 5);  
            if faces is not ():
                self.faceFound = True
                self.boundingBoxes_faces = [];
                for (x,y, w, h) in faces:
                    self.boundingBoxes_faces.append((x,y, x+w, y+h));
                    self.boundingBoxes_eyes[(x,y, x+w, y+h)] = [];
            else:
                self.faceFound = False;

    def getEyes_fullCasade(self, trackEyeBalls):
        # get the x and y gradients of the frame for tracking eyes balls
        frameGradientsXY = None;
        if trackEyeBalls:
            frameGradientsXYM = self.getFrameGradientXYM_eyeBallTracker(self.frame);
            

        if self.faceFound:
            for (x,y,x_w,y_h) in self.boundingBoxes_faces:
                roi_frame = self.frame[y:y_h, x:x_w];
                eyes = self.eyeCascade.detectMultiScale(roi_frame);
                if eyes is not None:
                    for (ex,ey,ew,eh) in eyes:
                        self.boundingBoxes_eyes[(x,y,x_w,y_h)].append((x+ex,y+ey,x+ex+ew,y+ey+eh));
                        if trackEyeBalls:
                            eye_loc = self.eyeBallTracker(self.frame, frameGradientsXYM, (x+ex,y+ey,x+ex+ew,y+ey+eh));
                            self.boundingBoxes_eyes[(x,y,x_w,y_h)].append(("eye_center",0,x+ex+eye_loc[0],y+ey+eye_loc[1]));    

    '''
        Calculate the gradient of the frame
        for the eye ball tracker
    '''
    def getFrameGradientXYM_eyeBallTracker(self,frame):
        img = cv2.equalizeHist(frame);
        #sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
        #sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
        sobelx = np.zeros(frame.shape);
        sobelx = np.asarray(cutils.cythonImageGradientX(img, img.shape[0], img.shape[1], sobelx));
        sobely = np.zeros(frame.shape);
        sobely = np.asarray(cutils.cythonImageGradientY(img, img.shape[0], img.shape[1], sobely));
        magnitude = np.zeros(frame.shape);
        magnitude = (np.asarray(cutils.cythonGetGradientMagnitude(sobelx, sobely, img.shape[0], img.shape[1], magnitude)));
        #magnitude = (sobelx * 2 + sobely * 2) ** 0.5;  
        return (sobelx, sobely, magnitude);
       
    
    def frameResize(self, frame, new_width = 500):
        h,w,c = 0,0,0;
        if len(frame.shape) > 2:
            h, w, c = frame.shape;
        else:
            h, w = frame.shape;
        ratio = float(new_width) / float(w);
        dim = (int(new_width), int(float(h) * ratio));
        frame = cv2.resize(frame, dim, interpolation= cv2.INTER_LINEAR);    
        return (ratio, frame);


    def eyeBallTracker(self, frame, frameGradientsXYM, eyeROIFrame):
        stdDevFactor = 25.0
        (x,y,x_w,y_h) = eyeROIFrame;
        (frameGradientsX, frameGradientsY, gradientMagnitude) = frameGradientsXYM;
        
        percentageEyeROI = 0.30;
        

        origX = x;
        origY = y;

        th = y_h - y;
        tw = x_w - x;
        
        y = y + th * percentageEyeROI;
        y_h = y_h - th * percentageEyeROI;
        x = x + tw * percentageEyeROI;
        x_w = x_w - tw * percentageEyeROI;

        y = int(y);
        x = int(x);
        y_h = int(y_h);
        x_w = int(x_w);

        changeX = x - origX;
        changeY = y - origY;

        eye = frame[y:y_h, x:x_w];
        

        (ratio, eye) = self.frameResize(eye, 20);
        (h,w) = eye.shape;
        (frameGradientsX, frameGradientsY, gradientMagnitude) = self.getFrameGradientXYM_eyeBallTracker(eye);
        egx = frameGradientsX;
        egy = frameGradientsY;
        egm = gradientMagnitude;
        eye = cv2.equalizeHist(eye);
        inv_eye = 255 - eye;
        
        #cv2.imshow("egx", egx);
        #cv2.imshow("egy", egy);
        #cv2.imshow("inv_eye", inv_eye);


        (meanGradMagn, stdGradMagn) = cv2.meanStdDev(egm);
        stdDev = stdGradMagn[0][0] / ((h*w) ** 0.5);
        dynamicThreshold = (stdDevFactor * stdDev + meanGradMagn[0][0]) % 255;
        # normalizing the gradient
        #dynamicThreshold = 15.0
        print "Dynamic Threshold + " + str(dynamicThreshold);

        #egx[egm <= dynamicThreshold] = 0.0;
        #egy[egm <= dynamicThreshold] = 0.0;
        #egx = np.divide(egx, egm);
        #egy = np.divide(egy, egm);
        temp = np.zeros(eye.shape);
        print egx.shape        
        egx = np.asarray(cutils.cythonNormailizeGradient(egx, h, w, dynamicThreshold, egm, temp));
        egy = np.asarray(cutils.cythonNormailizeGradient(egy, h, w, dynamicThreshold, egm, temp));

        outSum = np.zeros((h,w));
        outSum = cutils.cythonEyeBallTracker(inv_eye, outSum, egx, egy, egm);
        outSum = np.asarray(outSum);
        outSum = outSum / (outSum.shape[0] * outSum.shape[1]);
        (minVal, maxLoc, minLoc, maxLoc) =  cv2.minMaxLoc(outSum);
        inv_eye = cv2.circle(inv_eye, (maxLoc), 5, (255,0,0));
        maxLoc = (int(maxLoc[0]/ratio + changeX), int(maxLoc[1]/ratio + changeY));
        cv2.imshow("inv_eye", inv_eye);
        
        return maxLoc;
        




    def detectAllFaces(self, frame = None, trackEyeBalls = False):
        if frame is not None:
            self.setFrame(frame);
        else:
            self.getFrame();
        
        self.boundingBoxes_faces = [];
        self.boundingBoxes_eyes = {};
        if self.gotFrame:
            self.getFace_fullCascade();
            if trackEyeBalls:
                self.getEyes_fullCasade(trackEyeBalls);