import math

import numba
import numpy as np
import numpy.typing as npt


@numba.njit()
def vertex_index_buffer(h: int, w: int) -> npt.NDArray:
    """
    Each quad formed by 4 points can be split up in 2 triangles.
    returns a 2D array of height=(h-1)*(w-1)*2 and width 3. Each row corresponds to a triangle
    """
    N = (h - 1) * (w - 1) * 2
    n = 0
    triangle_vib = np.empty((N, 3), dtype=np.int32)

    # for each quadritlateral
    for y in range(h - 1):
        for x in range(w - 1):
            # indexes of the 4 points
            ind0 = y * w + x
            ind1 = ind0 + 1
            ind2 = ind0 + w
            ind3 = ind2 + 1

            # fill 2 triangles
            triangle_vib[n, :] = ind0, ind1, ind2
            triangle_vib[n + 1, :] = ind2, ind1, ind3
            n += 2
    return triangle_vib


@numba.jit(nopython=True)
def invert_map(xmap: npt.NDArray, ymap: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    h, w = xmap.shape
    xmap_inv = np.zeros_like(xmap) - 1
    ymap_inv = np.zeros_like(ymap) - 1

    triangle_vib = vertex_index_buffer(h, w)

    # for each triangle
    for k0, k1, k2 in triangle_vib:
        # get xy forrdinates of the triangles vertices
        x0 = xmap.ravel()[k0]
        x1 = xmap.ravel()[k1]
        x2 = xmap.ravel()[k2]

        y0 = ymap.ravel()[k0]
        y1 = ymap.ravel()[k1]
        y2 = ymap.ravel()[k2]

        # barycentric coordinates
        dy21 = y1 - y2
        dx20 = x0 - x2
        dx12 = x2 - x1
        dy20 = y0 - y2

        norm = dy21 * dx20 + dx12 * dy20

        i0 = k0 // w
        i1 = k1 // w
        i2 = k2 // w

        j0 = k0 % w
        j1 = k1 % w
        j2 = k2 % w

        # search area (rectangle surrounding current triangle)
        xmin = int(math.floor(min(x0, x1, x2)))
        ymin = int(math.floor(min(y0, y1, y2)))
        xmax = int(math.ceil(max(x0, x1, x2)))
        ymax = int(math.ceil(max(y0, y1, y2)))

        xmin = min(max(0, xmin), w - 1)
        ymin = min(max(0, ymin), h - 1)
        xmax = min(max(0, xmax), w - 1)
        ymax = min(max(0, ymax), h - 1)

        if abs(norm) <= 0.01:
            xmap_inv[ymin:ymax, xmin:xmax] = j0
            ymap_inv[ymin:ymax, xmin:xmax] = i0
            continue

        for px in range(xmin, xmax):
            pwx0 = dy21 * (px - x2)
            pwx1 = -dy20 * (px - x2)
            for py in range(ymin, ymax):
                # compute normalized weights of barycentric coordinates. Sum of weights must be 1
                w0 = (pwx0 + dx12 * (py - y2)) / norm
                w1 = (pwx1 + dx20 * (py - y2)) / norm
                w2 = 1 - w0 - w1

                # barycentric interpolation
                xmap_inv[py, px] = (j0 * w0 + j1 * w1 + j2 * w2)
                ymap_inv[py, px] = (i0 * w0 + i1 * w1 + i2 * w2)

    return xmap_inv, ymap_inv
