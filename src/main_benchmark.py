import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi

import iterative
import barycentric
import barycentric2
import map_generator


def measure_error(xmap, ymap, xmap_inv, ymap_inv):
    sh = xmap.shape
    assert ymap.shape == sh
    assert xmap_inv.shape == sh
    assert ymap_inv.shape == sh
    assert xmap.dtype == np.float32
    assert ymap.dtype == np.float32
    assert xmap_inv.dtype == np.float32
    assert ymap_inv.dtype == np.float32

    h, w = xmap.shape
    xmap_res, ymap_res = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

    # keep only values inside the image
    mask = (0 <= xmap.ravel()) & (xmap.ravel() < w) & (0 <= ymap.ravel()) & (ymap.ravel() < h)
    mask &= (0 <= xmap_inv.ravel()) & (xmap_inv.ravel() < w) & (0 <= ymap_inv.ravel()) & (ymap_inv.ravel() < h)

    xmap_t = cv2.remap(xmap, xmap_inv, ymap_inv, interpolation=cv2.INTER_LINEAR)
    ymap_t = cv2.remap(ymap, xmap_inv, ymap_inv, interpolation=cv2.INTER_LINEAR)

    square_errors = (xmap_t - xmap_res) ** 2 + (ymap_t - ymap_res) ** 2
    square_errors = square_errors.ravel()[mask]

    rmse = np.sqrt(np.mean(square_errors.ravel()))
    return rmse


def benchmark(map_gen, invert_map_function):
    print("Warming up")
    t0 = time.time()

    # generate_map = lambda N: map_generator.distortion_map(N)
    # generate_map = lambda N: map_generator.rot90(N)
    # generate_map = lambda N: map_generator.symmetry(N)
    # generate_map = lambda N: map_generator.zoom_out(N)

    generate_map = lambda N: map_gen(N)

    xmap, ymap = generate_map(4)

    res, _ = invert_map_function(xmap, ymap)
    dt = time.time() - t0
    print(f"Warmup time ({xmap.shape}, {xmap.dtype} -> {res.shape}, {res.dtype}): {dt:.3f} s")

    dt_list = []
    rmse_list = []
    ns = list(range(100, 1001, 100))
    for N in ns:
        xmap, ymap = generate_map(N)

        t0 = time.time()
        xmap_inv, ymap_inv = invert_map_function(xmap, ymap)
        dt = time.time() - t0
        dt_list.append(dt)

        rmse = measure_error(xmap, ymap, xmap_inv, ymap_inv)
        rmse_list.append(rmse)
        print(f"Computation time ({xmap.shape}, {xmap.dtype} -> {xmap_inv.shape}, {xmap_inv.dtype}): {dt:.3f} s")

        # img = generate_test_image(N)
        # warped = cv2.remap(img, xmap, ymap, cv2.INTER_LINEAR)
        # cv2.imshow("Warped", warped)
        # cv2.waitKeyEx(1)
        # xmap_inv, ymap_inv = invert_map_function(xmap, ymap)
        # unwarped = cv2.remap(warped, xmap_inv, ymap_inv, cv2.INTER_LINEAR)
        # cv2.imshow("Unwarped", unwarped)
        # cv2.waitKeyEx(0)

    return ns, dt_list, rmse_list


def main():
    # iterate over different map
    for map_gen in map_generator.get_list():
        dt_fig = plt.figure()
        dt_fig.suptitle(f"Computation time ({map_gen.__name__})")
        plt.xlabel("map size (width)")

        err_fig = plt.figure()
        err_fig.suptitle(f"Root Mean Square Error ({map_gen.__name__})")
        plt.xlabel("map size (width)")
        legends = []

        for algo in [iterative, barycentric, barycentric2]:
            ns, dts, rmse = benchmark(map_gen, algo.invert_map)
            dt_fig.gca().plot(ns, dts)
            err_fig.gca().plot(ns, rmse)
            legends.append(algo.__name__)

        for fig in [err_fig, dt_fig]:
            fig.gca().grid()
            fig.gca().legend(legends)
        plt.show()


if __name__ == '__main__':
    main()
