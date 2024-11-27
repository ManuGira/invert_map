import os
import pickle

import numpy as np
from scipy import ndimage as ndi


def distortion_map(N, sigma=None, force_generate=False):
    if sigma is None:
        sigma = max(1, N // 20)
    pickle_filename = f'./.cache/distortion_map_{N}_{sigma}.pickle'
    os.makedirs(os.path.dirname(pickle_filename), exist_ok=True)

    if not force_generate and os.path.exists(pickle_filename):
        with open(pickle_filename, 'rb') as handle:
            xmap, ymap = pickle.load(handle)
            return xmap, ymap

    sh = (N, N)
    t = np.random.normal(size=sh)
    dx = ndi.gaussian_filter(t, sigma, order=(0, 1))
    dy = ndi.gaussian_filter(t, sigma, order=(1, 0))
    dx *= 20 / dx.max()
    dy *= 20 / dy.max()
    yy, xx = np.indices(sh)
    xmap = (xx - dx).astype(np.float32)
    ymap = (yy - dy).astype(np.float32)

    with open(pickle_filename, 'wb') as handle:
        pickle.dump((xmap, ymap), handle, protocol=pickle.HIGHEST_PROTOCOL)
    return xmap, ymap


def rot90(N):
    rangeN = np.arange(N, dtype=np.float32)
    xmap, ymap = np.meshgrid(rangeN, rangeN)
    return ymap, N - 1 - xmap


def symmetry(N):
    rangeN = np.arange(N, dtype=np.float32)
    xmap, ymap = np.meshgrid(rangeN, rangeN)
    return N - 1 - xmap, ymap


def zoom_out(N):
    rangeN = np.arange(N, dtype=np.float32)
    xmap, ymap = np.meshgrid(rangeN, rangeN)
    xmap = (xmap - N / 2) * 2 + N / 2
    ymap = (ymap - N / 2) * 2 + N / 2
    return xmap, ymap


def zoom_in(N):
    rangeN = np.arange(N, dtype=np.float32)
    xmap, ymap = np.meshgrid(rangeN, rangeN)
    xmap = (xmap - N / 2) / 2 + N / 2
    ymap = (ymap - N / 2) / 2 + N / 2
    return xmap, ymap


def get_list():
    return [
        distortion_map,
        rot90,
        symmetry,
        zoom_out,
        zoom_in,
    ]
