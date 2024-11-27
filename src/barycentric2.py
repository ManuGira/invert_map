import numpy as np
import math

import numba


@numba.njit()
def vertex_index_buffer(h: int, w: int):
    N = (h - 1) * (w - 1)
    n = 0
    triangle_vib = np.empty((N, 4), dtype=np.int32)
    for y in range(h - 1):
        for x in range(w - 1):
            ind0 = y * w + x
            ind1 = ind0 + 1
            ind3 = ind0 + w
            ind2 = ind3 + 1
            triangle_vib[n, :] = ind0, ind1, ind2, ind3
            n += 1
    return triangle_vib


@numba.jit(nopython=True)
def invert_map(xmap, ymap, diagnostic=False):
    h, w = xmap.shape
    xmap_inv = np.zeros_like(xmap) - 1
    ymap_inv = np.zeros_like(ymap) - 1
    TN = np.zeros(shape=(h, w), dtype=np.int32)

    quad_vib = vertex_index_buffer(h, w)

    for k0, k1, k2, k3 in quad_vib:
        x0 = xmap.ravel()[k0]
        x1 = xmap.ravel()[k1]
        x2 = xmap.ravel()[k2]
        x3 = xmap.ravel()[k3]

        y0 = ymap.ravel()[k0]
        y1 = ymap.ravel()[k1]
        y2 = ymap.ravel()[k2]
        y3 = ymap.ravel()[k3]

        # barycentric coordinates
        # triangle 0,1,3
        dy31 = y1 - y3
        dx30 = x0 - x3
        dx13 = x3 - x1
        dy30 = y0 - y3

        # triangle 2,3,1
        dy13 = y3 - y1
        dx12 = x2 - x1
        dx31 = x1 - x3
        dy12 = y2 - y1

        # TODO: be carefull when norm is 0
        norm_013 = dy31 * dx30 + dx13 * dy30
        norm_231 = dy13 * dx12 + dx31 * dy12
        norm = norm_013 + norm_231

        i0 = k0 // w
        i1 = k1 // w
        i2 = k2 // w
        i3 = k3 // w

        j0 = k0 % w
        j1 = k1 % w
        j2 = k2 % w
        j3 = k3 % w

        # find surrounding rectangle
        xmin = int(math.floor(min(x0, x1, x2, x3)))
        ymin = int(math.floor(min(y0, y1, y2, y3)))
        xmax = int(math.ceil(max(x0, x1, x2, x3)))
        ymax = int(math.ceil(max(y0, y1, y2, y3)))

        xmin = min(max(0, xmin), w - 1)
        ymin = min(max(0, ymin), h - 1)
        xmax = min(max(0, xmax), w - 1)
        ymax = min(max(0, ymax), h - 1)

        if norm == 0:
            xmap_inv[ymin:ymax, xmin:xmax] = j0
            ymap_inv[ymin:ymax, xmin:xmax] = i0
            TN[ymin:ymax, xmin:xmax] += 1
            continue

        for px in range(xmin, xmax):
            pwx0_013 = dy31 * (px - x3)
            pwx1_013 = -dy30 * (px - x3)

            pwx2_231 = dy13 * (px - x1)
            pwx3_231 = -dy12 * (px - x1)
            for py in range(ymin, ymax):
                # compute non-normalized weights of barycentric coordinates
                w0_013 = (pwx0_013 + dx13 * (py - y3)) / norm_013
                w1_013 = (pwx1_013 + dx30 * (py - y3)) / norm_013
                w3_013 = 1 - w0_013 - w1_013

                is_valid_013 = (w0_013 >= 0 and w1_013 >= 0 and w3_013 >= 0) or (w0_013 <= 0 and w1_013 <= 0 and w3_013 <= 0)
                if is_valid_013:
                    val_x_013 = (j0 * w0_013 + j1 * w1_013 + j3 * w3_013)
                    val_y_013 = (i0 * w0_013 + i1 * w1_013 + i3 * w3_013)
                    xmap_inv[py, px] = val_x_013
                    ymap_inv[py, px] = val_y_013
                    TN[py, px] += 1
                    continue

                w2_231 = (pwx2_231 + dx31 * (py - y1)) / norm_231
                w3_231 = (pwx3_231 + dx12 * (py - y1)) / norm_231
                w1_231 = 1 - w2_231 - w3_231

                is_valid_231 = (w2_231 >= 0 and w3_231 >= 0 and w1_231 >= 0) or (w2_231 <= 0 and w3_231 <= 0 and w1_231 <= 0)
                if is_valid_231:
                    val_x_231 = (j2 * w2_231 + j3 * w3_231 + j1 * w1_231)
                    val_y_231 = (i2 * w2_231 + i3 * w3_231 + i1 * w1_231)

                    xmap_inv[py, px] = val_x_231
                    ymap_inv[py, px] = val_y_231
                    TN[py, px] += 1

    # if diagnostic:
    #     return xmap_inv, ymap_inv, TN
    # else:
    return xmap_inv, ymap_inv
