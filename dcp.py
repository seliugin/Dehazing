import numpy as np
import cv2
from cv2.ximgproc import guidedFilter
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.io import imsave, imread
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from skimage.color import rgb2hsv, hsv2rgb

# define filenames
IMG_FILEPATH = "haze2_red.png"
DST_FILEPATH = "haze2_result250.png"
IMG_NO_HAZE_FILEPATH = "haze1_no_cut.jpeg"
# dark channel window
DX = 15
DY = 15
# reg coeffs
K = 1
T0 = 0.01
# Ffilters params
GS = 10
R = 20
EPS = 0.01

# Q = 0.95
# P = 220
METRICS = {"SSIM": ssim, "MSE": mse, "PSNR": cv2.PSNR}


def get_dark_value(x, y, I, dx=7, dy=7):
    """Get minimal value through all the channels in considered window for one pixel by its coords

    Args:
        x, y (int) : pixel coordinates
        I (numpy ndarray): image, shape (x_size, y_size, 3)
        dx, dy (int, optional): window size. Defaults to (7, 7).

    Returns:
        int: dark value
    """
    r = np.min(I[x - dx : x + dx + 1, y - dy : y + dy + 1])
    return r


def get_dark_channel(I, dx, dy):
    """Get dark channel for an image

    Args:
        I (numpy ndarray): image, shape (x_size, y_size, 3)
        dx, dy (int): window size.

    Returns:
        numpy ndarray: image (x_size, y_size), where (x,y) - dark value for image I
    """
    x_size, y_size = I.shape[:2]
    res = np.pad(I, ((dx, dx), (dy, dy), (0, 0)), mode="edge")

    for x in range(dx, x_size + dx):
        for y in range(dy, y_size + dy):
            res[x, y] = get_dark_value(x, y, I, dx, dy)

    res = res[dx : x_size + dx, dy : y_size + dy, 0]
    return res


def window_min(I, dx, dy):
    """Window minimum filter for an image

    Args:
        I (numpy ndarray): image, shape (x_size, y_size)
        dx, dy (int): window size.

    Returns:
        numpy ndarray: image (x_size, y_size)
    """
    x_size, y_size = I.shape
    res = np.pad(I, ((dx, dx), (dy, dy)), mode="edge")
    for x in range(dx, x_size + dx):
        for y in range(dy, y_size + dy):
            res[x, y] = get_dark_value(x, y, I, dx, dy)

    res = res[dx : x_size + dx, dy : y_size + dy]
    return res


def jlong(I, dx=7, dy=7, k=0.9, t0=0.1, sigma=2, dc_quantile=0.999, p=200):
    """Single Remote Sensing Image Dehazing by
    J. Long et. al.: http://levir.buaa.edu.cn/publications/SingleRemoteSensingImageDehazing.pdf

    Args:
        I (numpy ndarray): image, shape (x_size, y_size, 3)
        dx, dy (int, optional): window size. Defaults to (7, 7).
        k (float, optional): reg param for dehazed image. Defaults to 0.9.
        t0 (float, optional): reg param for transmission map. Defaults to 0.1.
        sigma (int, optional): sigma for gaussian filter. Defaults to 2.
        dc_quantile (float, optional): quantile for veil color estimation. Defaults to 0.999.
        p (int, optional): color correction. Defaults to 200.

    Returns:
        numpy ndarray: dehazed image
    """

    # get veil color
    I_dark = get_dark_channel(I, dx, dy)
    q_I = np.quantile(I_dark, dc_quantile)
    a_rgb = (I[(I_dark >= q_I)]).mean(axis=0)
    A = a_rgb * np.ones(I.shape)
    print("Long Цвет дымки:", a_rgb)
    I_a = I / A
    I_a = (I_a - I_a.min()) / (I_a.max() - I_a.min())

    # get and filter veil and transmission
    V = get_dark_channel(I_a, 1, 1).astype(np.float32)
    V_filtered = gaussian_filter(V, sigma)
    assert V.min() >= 0 and V.max() <= 1

    t = 1 - V_filtered
    t = np.dstack((t, t, t))

    # color correction
    M = np.ones(t.shape) * p
    diff = M / np.abs(I_a - A)
    print(f"diff: min={diff.min()}, max={diff.max()}")
    t = np.minimum(np.maximum(diff, 1) * t, 1)
    #     t = np.minimum(np.maximum(M / np.abs(I - A), 1) * t, 1)

    # final result
    J = k * A + (I - k * A) / np.maximum(t, np.full(t.shape, t0))
    J = np.clip(J, 0, 255).astype(np.uint8)
    return J


def idcp(I, dx=7, dy=7, k=0.95, t0=0.1, r=30, eps=0.01, dc_quantile=0.999):
    """Dark Channel Prior algorithm using guided filter
    K. He: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.672.3815&rep=rep1&type=pdf


    Args:
        I (numpy ndarray): image, shape (x_size, y_size, 3)
        dx, dy (int, optional): window size. Defaults to (7, 7).
        k (float, optional): reg param for dehazed image. Defaults to 0.95.
        t0 (float, optional): reg param for transmission map. Defaults to 0.1.
        r (int, optional): guided filter radius. Defaults to 30.
        eps (float, optional): reg param for guided filter. Defaults to 0.01.
        dc_quantile (float, optional): quantile for veil color estimation. Defaults to 0.999.

    Returns:
        numpy ndarray: dehazed image
    """
    # get veil color
    I_dark = get_dark_channel(I, dx, dy)
    q_I_dark = np.quantile(I_dark, dc_quantile)
    I_intens = I.sum(axis=2)
    q_intens = np.quantile(I_intens[I_dark >= q_I_dark], 0.9)
    a_rgb = (I[(I_dark >= q_I_dark) & (I_intens >= q_intens)]).mean(axis=0)
    A = a_rgb * np.ones(I.shape)
    print("IDCP Цвет дымки:", a_rgb)

    # get and filter transmisson map
    I_a = I / A
    V = get_dark_channel(I_a, dx, dy).astype(np.float32)
    t = 1 - k * V
    t = guidedFilter(I, t, r, eps)
    t = np.dstack((t, t, t))

    # final dehazed image
    J = (I - A) / np.maximum(t, np.full(t.shape, t0)) + A
    J = np.clip(J, 0, 255).astype(np.uint8)
    return J


def zhu_depth_estim(I, dx=7, dy=7, r=30, eps=0.01, gf_on=True):
    """Atmospheric Light Estimation Based Remote Sensing Image Dehazing by
    Z. Zhu et. al.: https://www.mdpi.com/2072-4292/13/13/2432/htm
    Depth map estimation

    Args:
        I (numpy ndarray): image, shape (x_size, y_size, 3)
        dx, dy (int, optional): window size. Defaults to (7, 7).
        r (int, optional): guided filter radius. Defaults to 30.
        eps (float, optional): reg param for guided filter. Defaults to 0.01.
        gf_on (bool, optional): apply guided filter or not. Defaults to True.

    Returns:
        numpy ndarray: depth map (x_size, y_size)
    """
    I_hsv = rgb2hsv(I)
    w0 = 0.172066
    w1 = 1.108955
    w2 = -0.952585

    d = (w0 + w1 * I_hsv[:, :, 2] + w2 * I_hsv[:, :, 1]).astype(np.float32)
    #     d = np.dstack((d,d,d)) #DCP по сути и есть оконный минимум
    d = window_min(d, dx, dy)

    if gf_on:
        d = guidedFilter(I, d, r, eps)

    return d


def zhu_depth(
    I, dx=7, dy=7, t0=0.01, r=30, eps=0.01, d_quantile=0.999, gf_on=True, beta=1.2
):
    """Atmospheric Light Estimation Based Remote Sensing Image Dehazing by
    Z. Zhu et. al.: https://www.mdpi.com/2072-4292/13/13/2432/htm
    Dehazing algorithm

    Args:
        I (numpy ndarray): image, shape (x_size, y_size, 3)
        dx, dy (int, optional): window size. Defaults to (7, 7).
        t0 (float, optional): reg param for transmission map. Defaults to 0.1.
        r (int, optional): guided filter radius. Defaults to 30.
        eps (float, optional): reg param for guided filter. Defaults to 0.01.
        d_quantile (float, optional): quantile for veil color estimation. Defaults to 0.999.
        gf_on (bool, optional): apply guided filter or not. Defaults to True.
        beta (float, optional): absorption coefficient. Defaults to 1.2

    Returns:
        numpy ndarray: depth map (x_size, y_size)
    """

    d = zhu_depth_estim(I, dx, dy, r, eps, gf_on)

    q_d = np.quantile(d, d_quantile)
    I_intens = I.sum(axis=2)

    I_max = I_intens[d >= q_d].max()
    a_rgb = I[(d >= q_d) & (I_intens == I_max)].mean(axis=0)
    print("Zhu depth map цвет дымки:", a_rgb)

    A = a_rgb * np.ones(I.shape)

    t = np.exp(-beta * d)
    t = np.dstack((t, t, t))

    J = (I - A) / np.maximum(t, np.full(t.shape, t0)) + A
    J = np.clip(J, 0, 255).astype(np.uint8)
    return J


def zhu_dcp(
    I,
    dx=7,
    dy=7,
    k=0.95,
    t0=0.01,
    r=30,
    eps=0.01,
    d_quantile=0.999,
    gf_on=True,
    a_mean=True,
):
    """Atmospheric Light Estimation Based Remote Sensing Image Dehazing by
    Z. Zhu et. al.: https://www.mdpi.com/2072-4292/13/13/2432/htm
    and Dark Channel Prior dehazing algorithm

    Args:
        I (numpy ndarray): image, shape (x_size, y_size, 3)
        dx, dy (int, optional): window size. Defaults to (7, 7).
        k (float, optional): reg param for dehazed image. Defaults to 0.95.
        t0 (float, optional): reg param for transmission map. Defaults to 0.1.
        r (int, optional): guided filter radius. Defaults to 30.
        eps (float, optional): reg param for guided filter. Defaults to 0.01.
        d_quantile (float, optional): quantile for veil color estimation. Defaults to 0.999.
        gf_on (bool, optional): apply guided filter or not. Defaults to True.
        a_mean (bool, optional): average to estimate veil color or not . Defaults to True.

    Returns:
        numpy ndarray: depth map (x_size, y_size)
    """

    d = zhu_depth_estim(I, dx, dy, r, eps, gf_on)

    q_d = np.quantile(d, d_quantile)
    I_intens = I.sum(axis=2)

    if a_mean:  # усреднение по пикселям
        a_rgb = (I[d >= q_d]).mean(axis=0)
    else:  # самый яркий пиксель
        I_max = I_intens[d >= q_d].max()
        a_rgb = I[(d >= q_d) & (I_intens == I_max)].mean(axis=0)
    A = a_rgb * np.ones(I.shape)
    print(f"Zhu: mean={a_mean}, gf={gf_on} цвет дымки:", a_rgb)

    I_a = I / A
    V = get_dark_channel(I_a, dx, dy).astype(np.float32)
    t = 1 - k * V
    t = guidedFilter(I, t, r, eps)
    #     t = 1 - k * (1 - t)
    t = np.dstack((t, t, t))

    J = (I - A) / np.maximum(t, np.full(t.shape, t0)) + A
    J = np.clip(J, 0, 255).astype(np.uint8)
    return J


# reddish color veil generator
def a_generator():
    h = np.random.rand() / 6
    s = np.random.rand() / 2
    v = np.random.rand() / 2 + 0.5
    a_hsv = [h, s, v]
    a_rgb = hsv2rgb(a_hsv) * 255
    return a_rgb


# synthetic haze generator
def hazing(
    img_no_haze_filepath,
    dst_filepath,
    A_color=[250.0, 220.0, 220.0],
    t_min=0,
    t_max=0.8,
    uniform=False,
):
    """Haze generator

    Args:
        img_no_haze_filepath (string): clear image filename
        dst_filepath (string): filename for hazed image
        A_color (list, optional): veil color. Defaults to [250.0, 220.0, 220.0].
        t_min, t_max (float, optional): minimum and maximum transmission. Defaults to 0 and 0.8.
        uniform (bool, optional): uniform or gradient veil. Defaults to False.
    """
    image = imread(img_no_haze_filepath)

    if uniform:
        t = np.tile((t_min + t_max) / 2, (image.shape[0], image.shape[1], 1))
    else:
        t_1 = np.linspace(t_min, t_max, image.shape[0])
        t = np.tile(t_1, (image.shape[1], 1))
        t = t.T

    t = np.dstack([t, t, t])
    print("Цвет дымки:", A_color)
    A = np.array(A_color) * np.ones(image.shape)

    I = image * t + A * (np.ones(image.shape) - t)
    imsave(dst_filepath, I.astype(np.uint8))
    return 0


METRICS = {"SSIM": ssim, "MSE": mse, "PSNR": cv2.PSNR}


def calculate_metrics(orig, img, metrics=METRICS):
    """Quality metrics calculator

    Args:
        orig (numpy ndarray): clear image
        img (numpy ndarray): dehazed image
        metrics (dict, optional): metrics functions. Defaults to METRICS.

    Returns:
        list: metrics values
    """
    m_vals = []

    for m in metrics:
        if m == "SSIM":
            m_vals.append(np.around(METRICS[m](orig, img, channel_axis=2), decimals=3))
            continue
        m_vals.append(np.around(METRICS[m](orig, img), decimals=2))
    return m_vals
