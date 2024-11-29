import numpy as np
import numba
import numpy.typing as npt
import math


# https://stackoverflow.com/a/65566295
@numba.jit(nopython=True)
def bilinear_inverse(p, s, vertices, numiter=4):
    """
    Compute the inverse of the bilinear map from the unit square
    [(0,0), (1,0), (1,1), (0,1)]
    to the quadrilateral vertices = [p0, p1, p2, p3]

    Parameters:
    ----------
    p: array of shape (2,).
    vertices: array of shape (4, 2).
    numiter: Number of Newton iterations.

    Returns:
    --------
    s: array of shape (2,).
    """
    p = np.asarray(p)
    v = np.asarray(vertices)

    si, sj = s[0], s[1]

    for k in range(numiter):
        # Residual
        ri = (v[0, 0] * (1 - si) * (1 - sj) + v[1, 0] * si * (1 - sj) + v[2, 0] * si * sj + v[3, 0] * (1 - si) * sj - p[0])
        rj = (v[0, 1] * (1 - si) * (1 - sj) + v[1, 1] * si * (1 - sj) + v[2, 1] * si * sj + v[3, 1] * (1 - si) * sj - p[1])

        # Jacobian
        J11 = (-v[0, 0] * (1 - sj) + v[1, 0] * (1 - sj) + v[2, 0] * sj - v[3, 0] * sj)
        J21 = (-v[0, 1] * (1 - sj) + v[1, 1] * (1 - sj) + v[2, 1] * sj - v[3, 1] * sj)
        J12 = (-v[0, 0] * (1 - si) - v[1, 0] * si + v[2, 0] * si + v[3, 0] * (1 - si))
        J22 = (-v[0, 1] * (1 - si) - v[1, 1] * si + v[2, 1] * si + v[3, 1] * (1 - si))

        detJ = J11 * J22 - J12 * J21
        inv_detJ = 1. / detJ

        si -= inv_detJ * (J22 * ri - J12 * rj)
        sj -= inv_detJ * (-J21 * ri + J11 * rj)

    return s


@numba.njit()
def vertex_index_buffer(h: int, w: int):
    N = (h - 1) * (w - 1)
    n = 0
    quad_vib = np.empty((N, 4), dtype=np.int32)
    for y in range(h - 1):
        for x in range(w - 1):
            ind0 = y * w + x
            ind1 = ind0 + 1
            ind3 = ind0 + w
            ind2 = ind3 + 1
            quad_vib[n, :] = ind0, ind1, ind2, ind3
            n += 1
    return quad_vib



@numba.jit(nopython=True)
def invert_map(xmap: npt.NDArray, ymap: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    h, w = xmap.shape
    xmap_inv = np.zeros_like(xmap) - 1
    ymap_inv = np.zeros_like(ymap) - 1

    quad_vib = vertex_index_buffer(h, w)

    # for each quadtrilateral
    for k0, k1, k2, k3 in quad_vib:
        # get xy forrdinates of the triangles vertices
        x0 = xmap.ravel()[k0]
        x1 = xmap.ravel()[k1]
        x2 = xmap.ravel()[k2]
        x3 = xmap.ravel()[k3]

        y0 = ymap.ravel()[k0]
        y1 = ymap.ravel()[k1]
        y2 = ymap.ravel()[k2]
        y3 = ymap.ravel()[k3]

        vertices = np.array([
            [y0, x0],
            [y1, x1],
            [y2, x2],
            [y3, x3],
        ], dtype=np.float32)

        i0 = k0 // w
        i1 = k1 // w
        i2 = k2 // w
        i3 = k3 // w

        j0 = k0 % w
        j1 = k1 % w
        j2 = k2 % w
        j3 = k3 % w

        s = np.empty((2,), dtype=np.float32)
        s[0] = (i0+i1+i2+i3)/4
        s[1] = (j0+j1+j2+j3)/4

        # search area (rectangle surrounding current triangle)
        xmin = min(x0, x1, x2, x3)
        ymin = min(y0, y1, y2, y3)
        xmax = max(x0, x1, x2, x3)
        ymax = max(y0, y1, y2, y3)

        # if max(xmax-xmin, ymax-ymin) < 0.01:
        #     xmap_inv[ymin, xmin] = j0
        #     ymap_inv[ymin, xmin] = i0
        #     continue

        xmin = int(math.floor(xmin))
        ymin = int(math.floor(ymin))
        xmax = int(math.ceil(xmax))
        ymax = int(math.ceil(ymax))

        xmin = min(max(0, xmin), w - 1)
        ymin = min(max(0, ymin), h - 1)
        xmax = min(max(0, xmax), w - 1)
        ymax = min(max(0, ymax), h - 1)


        for px in range(xmin, xmax):
            for py in range(ymin, ymax):
                # Start in the center

                p = np.array([py, px], dtype=np.float32)

                iy, ix = bilinear_inverse(p, s, vertices, numiter=4)

                # barycentric interpolation
                ymap_inv[py, px] = iy
                xmap_inv[py, px] = ix

    return xmap_inv, ymap_inv