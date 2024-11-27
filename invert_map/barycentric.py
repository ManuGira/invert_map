import numpy as np
import math

import numba


@numba.njit()
def vertex_index_buffer(h: int, w: int):
    N = (h - 1) * (w - 1) * 2
    n = 0
    triangle_vib = np.empty((N, 3), dtype=np.int32)
    for y in range(h - 1):
        for x in range(w - 1):
            i0 = y * w + x
            i1 = i0 + 1
            i2 = i0 + w
            i3 = i2 + 1
            triangle_vib[n, :] = i0, i1, i2
            triangle_vib[n + 1, :] = i2, i1, i3
            n += 2
    return triangle_vib


@numba.jit(nopython=True)
def invert_map(xmap, ymap, diagnostic=False):
    h, w = xmap.shape
    xmap_inv = np.zeros_like(xmap) - 1
    ymap_inv = np.zeros_like(ymap) - 1
    TN = np.zeros(shape=(h, w), dtype=np.int32)

    triangle_vib = vertex_index_buffer(h, w)

    for k0, k1, k2 in triangle_vib:
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
        # TODO: be carefull when norm is 0
        norm = dy21 * dx20 + dx12 * dy20

        i0 = k0 // w
        i1 = k1 // w
        i2 = k2 // w

        j0 = k0 % w
        j1 = k1 % w
        j2 = k2 % w

        # find surrounding rectangle
        xmin = int(math.floor(min(x0, x1, x2)))
        ymin = int(math.floor(min(y0, y1, y2)))
        xmax = int(math.ceil(max(x0, x1, x2)))
        ymax = int(math.ceil(max(y0, y1, y2)))

        xmin = min(max(0, xmin), w-1)
        ymin = min(max(0, ymin), h-1)
        xmax = min(max(0, xmax), w-1)
        ymax = min(max(0, ymax), h-1)

        if norm == 0:
            xmap_inv[ymin:ymax, xmin:xmax] = j0
            ymap_inv[ymin:ymax, xmin:xmax] = i0
            TN[ymin:ymax, xmin:xmax] += 1
            continue

        for px in range(xmin, xmax):
            pwx0 = dy21 * (px - x2)
            pwx1 = -dy20 * (px - x2)
            for py in range(ymin, ymax):
                # compute non-normalized weights of barycentric coordinates
                w0 = (pwx0 + dx12 * (py - y2)) / norm
                w1 = (pwx1 + dx20 * (py - y2)) / norm
                w2 = 1 - w0 - w1

                # if w0 < 0 or w1 < 0 or w2 < 0:
                #     continue

                xmap_inv[py, px] = (j0 * w0 + j1 * w1 + j2 * w2)
                ymap_inv[py, px] = (i0 * w0 + i1 * w1 + i2 * w2)
                TN[py, px] += 1

    # if diagnostic:
    #     return xmap_inv, ymap_inv, TN
    # else:
    return xmap_inv, ymap_inv
