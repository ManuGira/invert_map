import cv2

import numpy as np
from scipy import ndimage as ndi

import map_generator
import barycentric
import barycentric2
import iterative
import bilinear


def generate_test_image(N):
    sh = (N, N)
    img = np.zeros(sh)
    img[::10, :] = 1
    img[:, ::10] = 1

    p0 = N // 10
    size = N * 8 // 10
    th = 10
    img[p0:p0 + size, p0:p0 + th] = 1
    img[p0:p0 + th, p0:p0 + size * 2 // 3] = 1
    img[p0 + size // 2:p0 + size // 2 + th, p0:p0 + size // 2] = 1

    img = ndi.gaussian_filter(img, 0.5)
    return img


def main(algo):
    N = 500
    xmap, ymap = map_generator.distortion_map(N)
    # xmap, ymap = map_generator.rot90(N)
    # xmap, ymap = map_generator.symmetry(N)
    # xmap, ymap = map_generator.zoom_out(N)

    # xmap, ymap = np.meshgrid(range(N), range(N))
    # xmap = cv2.rotate(xmap, cv2.ROTATE_90_CLOCKWISE).astype(np.float32)
    # ymap = cv2.rotate(ymap, cv2.ROTATE_90_CLOCKWISE).astype(np.float32)

    img = generate_test_image(N)
    warped = cv2.remap(img, xmap, ymap, cv2.INTER_LINEAR)

    cv2.imshow("Warped", warped)
    cv2.waitKeyEx(1)

    xmap_inv, ymap_inv = algo.invert_map(xmap, ymap)
    unwarped = cv2.remap(warped, xmap_inv, ymap_inv, cv2.INTER_LINEAR)

    cv2.imshow(f"Unwarped with {algo.__name__} algo", unwarped)
    cv2.waitKeyEx(0)


if __name__ == '__main__':
    main(barycentric)
    main(barycentric2)
    main(iterative)
    main(bilinear)
