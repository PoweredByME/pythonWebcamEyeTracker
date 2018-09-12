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


'''
import cython;

@cython.boundscheck(False)
cpdef double[:,:] cythonEyeBallTracker(unsigned char [:, :] frame,
                                        double[:,:] outSum,
                                        double[:,:] egx,
                                        double[:,:] egy,
                                        double[:,:] egm
                                        ):
    cdef int x,y,w,h;
    h = frame.shape[0];
    w = frame.shape[1];
    for y in range(0,h):
        for x in range(0,w):
            if egx[y,x] == 0.0 and egy[y,x] == 0.0:
                continue;
            outSum = testPossibleCenterFormulae(x,y,frame, egx[y,x], egy[y,x], h, w, outSum);
    return outSum;

cpdef double[:,:] testPossibleCenterFormulae(int x, int y, unsigned char[:,:] inv_eye, double egx, double egy, int h, int w, double[:,:] outSum):
    cdef double dx, dy, gx, gy, dotProduct, dm;
    cdef int cx, cy;
    gx = egx;
    gy = egy;
    for cy in range(h):
        for cx in range(w):
            if x == cx and y == cy:
                continue;
            dx = - cx + x;
            dy = - cy + y;
            dm = dx * 2 + dy * 2;
            dm = dm ** 0.5;
            if dm == 0.0:
                dx = 0.0;
                dy = 0.0;
            else:
                dx = dx / dm;
                dy = dy / dm
            dotProduct = max(0.0, dx*gx + dy*gy);
            outSum[y,x] = outSum[y,x] + dotProduct * dotProduct * inv_eye[cy,cx];
    return outSum;    


cpdef double[:,:] cythonImageGradientX(char[:,:] frame, int frameHeight, int frameWidth, double[:,:] out):
    
cpdef double[:,:] cythonImageGradientY(char[:,:] frame, int frameHeight, int frameWidth, double[:,:] out):
    h = frameHeight;
    w = frameWidth;
    for x in range(w):
        out[0,x] = frame[1,x] - frame[0,x];
        for y in range(1, h-1):
            out[y,x] = (frame[y+1, x] - frame[y-1,x]) / 2;
        out[h-1, x] = frame[h-1,x] - frame[h-2,x];
    return out;

cpdef double[:,:] cythonGetGradientMagnitude(double[:,:] gradX, double[:,:] gradY, int h, int w, double[:,:] out):
    for y in range(h):
        for x in range(w):
            out[y,x] = (gradX[y,x] * 2 + gradY[y,x] * 2) ** 0.5;
    return out;

cpdef double[:,:] cythonNormailizeGradient(double[:,:] gradient, int h, int w, double magThresh, double[:,:]mag, double[:,:] out):
    for y in range(h):
        for x in range(w):
            g = gradient[y,x];
            m = mag[y,x];
            if m == 0 and g == 0:
                out[y,x] = 0.0;
                continue;
            if m < magThresh or m == 0.0:
                out[y,x] = 0.0;
                continue;
            out[y,x] = g / m;
    return out;
'''