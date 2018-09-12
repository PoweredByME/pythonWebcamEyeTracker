import cython;

cpdef double[:,:] calcGradientX(unsigned char [:,:] frame, int frameHeight, int frameWidth, double[:,:] out):
    h = frameHeight;
    w = frameWidth;
    for y in range(h):
        out[y,0] = frame[y,1] - frame[y,0];
        for x in range(1,w-1):
            out[y,x] = (frame[y,x+1] - frame[y,x-1]) / 2.0;
        out[y, w-1] = frame[y, w-1] - frame[y, w-2];
    return out;


cpdef double[:,:] testPossibleCenterFormula(int x,
                                            int y,
                                            unsigned char [:,:] inv_eye,
                                            double gX,
                                            double gY,
                                            double [:,:] out,
                                            int h,
                                            int w):
    
    cdef double dx = 0.0, dy = 0.0, magn = 0.0, dotProduct = 0.0;

    for cy in range(0,h):
        for cx in range(0,w):
            if cx == x and cy == y:
                continue;
            dx = x - cx;
            dy = y - cy;
            # normalize d
            magn = dx ** 2 + dy ** 2;
            magn = magn ** 0.5;
            if magn == 0.0:
                dx = 0.0;
                dy = 0.0;
            else:
                dx /= magn;
                dy /= magn;
            dotProduct = dx*gX + dy*gY;
            dotProduct = max(0.0, dotProduct);
            
            out[cy,cx] += (dotProduct ** 2) * inv_eye[cy,cx];
    
    return out;

'''
    This method used for calculating the point of the pupil is inspired from the
    method by Tristan Hume.
    https://github.com/trishume/eyeLike
'''
cpdef double[:,:] findEyeCenter(unsigned char [:,:] inv_eye,
                                double[:,:] gx,
                                double[:,:] gy,
                                double[:,:] magn,
                                int frameHeight,
                                int frameWidth,
                                double[:,:] out):
    h = frameHeight;
    w = frameWidth;
    for y in range(0,h):
        for x in range(0,w):
            gX = gx[y,x];
            gY = gy[y,x];
            if gX == 0.0 and gY == 0.0:
                continue;
            out = testPossibleCenterFormula(x, y, inv_eye, gX, gY, out, h, w);
    
    return out;
