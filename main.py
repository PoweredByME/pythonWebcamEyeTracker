import camera, display, utils;



def myFrameFunction(cam, disp):
    ''' 
        This function capture a frame from the video stream and
        do required processing on it and then display it.
    '''
    ret, frame = cam.getVideoFrame();
    if not ret:
        print "No frame available. Ending video stream";

    

    disp.showFrame(frame);
            






def main():
    cam = camera.Camera();
    cam.setVideoSource(0);
    disp = display.Display();

    while cam.isVideoSourceAvaiable():
        myFrameFunction(cam, disp);
        if utils.CV_isButtonPressed("q"):
            break;

    cam.end();
    disp.end();




if __name__  == "__main__":
    main();