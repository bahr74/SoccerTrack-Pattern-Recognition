"""Member 1 NumPy HOG and optional HSV features (RGB uint8 patches)."""
from __future__ import annotations
import numpy as np
import cv2

HOG_DIM_64x32 = 756


def _check(X):
    if not isinstance(X, np.ndarray) or X.dtype != np.uint8 or X.ndim != 4 or X.shape[1:] != (64, 32, 3):
        raise ValueError('Expected RGB uint8 [N,64,32,3]')


def hog_features(patches_rgb, cfg):
    """Bilinear spatial/orientation votes, unsigned 9 bins, 8px cells, L2-Hys.

    Ordering: block y,x; cell y,x; orientation. OpenCV uses block/cell x,y.
    Unlike OpenCV this educational descriptor has no per-block Gaussian weighting.
    Consequently descriptors agree geometrically, but are not bit-identical.
    """
    _check(patches_rgb)
    hcfg = cfg['hog']
    if tuple(hcfg['cell']) != (8,8) or tuple(hcfg['block']) != (2,2) or hcfg['bins'] != 9:
        raise ValueError('Phase 2 HOG requires 8x8 cells, 2x2 blocks and 9 bins')
    result = np.empty((len(patches_rgb), 756), dtype=np.float32)
    yy, xx = np.indices((64,32))
    cy, cx = (yy + .5)/8 - .5, (xx + .5)/8 - .5
    iy, ix = np.floor(cy).astype(int), np.floor(cx).astype(int)
    for n, patch in enumerate(patches_rgb):
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY).astype(np.float32)
        padded = np.pad(gray, 1, mode='reflect')
        gx = padded[1:-1,2:] - padded[1:-1,:-2]
        gy = padded[2:,1:-1] - padded[:-2,1:-1]
        magnitude = np.hypot(gx,gy)
        angle = (np.degrees(np.arctan2(gy,gx)) % 180)/20 - .5
        ib = np.floor(angle).astype(int)
        hist = np.zeros((8,4,9),dtype=float)
        for dy in (0,1):
            for dx in (0,1):
                sy, sx = iy+dy, ix+dx
                spatial = (1-np.abs(cy-sy))*(1-np.abs(cx-sx))
                valid = (sy>=0)&(sy<8)&(sx>=0)&(sx<4)
                for db in (0,1):
                    weight = magnitude*spatial*(1-np.abs(angle-(ib+db)))
                    np.add.at(hist,(sy[valid],sx[valid],(ib[valid]+db)%9),weight[valid])
        blocks=[]
        for y in range(7):
            for x in range(3):
                v=hist[y:y+2,x:x+2].ravel()
                v=v/np.sqrt(np.dot(v,v)+1e-6)
                v=np.minimum(v,.2)
                blocks.append(v/np.sqrt(np.dot(v,v)+1e-6))
        result[n]=np.concatenate(blocks)
    return result


def hog_glyph(descriptor, cfg):
    """Draw block-averaged orientation votes on a 64x32 float32 glyph."""
    d=np.asarray(descriptor)
    if d.shape != (756,) or not np.isfinite(d).all():
        raise ValueError('Expected a finite 756-dimensional descriptor')
    hist=np.zeros((8,4,9)); counts=np.zeros((8,4,1))
    blocks=d.reshape(7,3,2,2,9)
    for y in range(7):
        for x in range(3):
            hist[y:y+2,x:x+2]+=blocks[y,x]; counts[y:y+2,x:x+2]+=1
    hist/=np.maximum(counts,1)
    image=np.zeros((64,32),dtype=np.float32)
    for y in range(8):
        for x in range(4):
            for b in range(9):
                a=np.deg2rad((b+.5)*20); dx=3*np.cos(a); dy=3*np.sin(a)
                cv2.line(image,(round(x*8+4-dx),round(y*8+4-dy)),
                         (round(x*8+4+dx),round(y*8+4+dy)),float(hist[y,x,b]),1)
    return image


def color_histograms(patches_rgb, bins):
    _check(patches_rgb)
    if len(bins)!=3 or any(not isinstance(v,int) or v<1 for v in bins):
        raise ValueError('bins must be three positive integers')
    result=np.zeros((len(patches_rgb),int(np.prod(bins))),dtype=np.float32)
    for i,patch in enumerate(patches_rgb):
        hsv=cv2.cvtColor(patch,cv2.COLOR_RGB2HSV)
        hist=cv2.calcHist([hsv],[0,1,2],None,list(bins),[0,180,0,256,0,256]).ravel()
        result[i]=hist/max(hist.sum(),1)
    return result
