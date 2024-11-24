import cv2
import numpy as np
import numpy.typing as npt

def invert_map(xmap: npt.NDArray, ymap: npt.NDArray, max_iter=20):
    F = np.stack((xmap, ymap), axis=2)
    sh = F.shape[:2]
    I = np.zeros_like(F)
    I[:, :, 1], I[:, :, 0] = np.indices(sh)
    P = np.copy(I)
    damping = 0.9
    for i in range(max_iter):
        error = cv2.remap(F, P, None, interpolation=cv2.INTER_LINEAR) - I
        P -= error * damping

    xmap_inv, ymap_inv = P[:, :, 0], P[:, :, 1]
    return xmap_inv, ymap_inv
