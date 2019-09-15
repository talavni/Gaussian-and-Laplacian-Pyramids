import numpy as np
import scipy.ndimage as sp
from scipy import signal
import matplotlib.pyplot as plt
import copy
from skimage.io import imread


def build_filter_vec(filter_size):
    """
    returns a row vector shape (1, filter_size)
    """
    filter_vec = [[1]]
    conv = [[1, 1]]
    div = np.power(2, filter_size - 1)
    for i in range(filter_size - 1):
        filter_vec = signal.convolve2d(filter_vec, conv, mode="full")
    return filter_vec / div


def reduce_image(im):
    return im[::2, ::2]


def blur(im, filter_vec):
    """
    blur image
    :param im: an ndarray
    :param filter_vec: the filter for blurring
    :return: blurred image
    """
    temp = sp.filters.convolve(im, filter_vec)
    return sp.filters.convolve(temp, np.transpose(filter_vec))


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    builds guassian pyramid out of the image
    :param im: ndarray
    :param max_levels: num of levels for the pyramid
    :param filter_size: the size of filter for blurring
    :return: all levels of the pyramid and the filter vector
    """
    filter_vec = build_filter_vec(filter_size)
    pyr = [im]
    for i in range(1, max_levels):
        temp = blur(pyr[i-1], filter_vec)
        pyr.append(reduce_image(temp))
    return pyr, filter_vec


def zero_pad(im):
    """
    pad an image with zeros
    :param im: ndarray
    :return: padded image
    """
    padded_im = np.zeros((im.shape[0]*2, im.shape[1]*2))
    padded_im[::2, ::2] = im
    return padded_im


def expand(im, filter_vec):
    """
    expand an image by 2
    :param im: ndarray
    :param filter_vec: filter for blurring
    :return:
    """
    padded_im = np.zeros((im.shape[0] * 2, im.shape[1] * 2))
    padded_im[::2, ::2] = im
    return blur(padded_im, 2 * filter_vec)


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    build a laplacian pyramid for the image
    :param im: ndarray
    :param max_levels: num of levels for the pyramid
    :param filter_size: size of filter for blurring
    :return: all levels of the pyramid and the filter vector
    """
    filter_vec = build_filter_vec(filter_size)
    pyr = []
    gaussian = build_gaussian_pyramid(im, max_levels, filter_size)[0]
    for i in range(1, max_levels):
        temp = expand(gaussian[i], filter_vec)
        temp = gaussian[i-1] - temp
        pyr.append(temp)
    pyr.append(gaussian[max_levels - 1])
    return pyr, filter_vec


def laplacian_to_image(lpyr, filter_vec, coeff):
    """
    reconstructs image from laplacian pyramid
    :param lpyr: laplacian pyramid
    :param filter_vec: filter vector
    :param coeff: size of the level of the pyramid. multiply each level of pyramid by corresponding coeff
    :return: image
    """
    lpyr = lpyr * np.array(coeff)
    im = lpyr[-1]
    for i in range(len(lpyr) - 2, -1, -1):
        im = expand(im, filter_vec) + lpyr[i]
    return im


def stretch(im):
    """
    stretch values of image between 0 and 1
    :param im: ndarray
    :return: stretched image
    """
    mini = min(im.flatten())
    maxi = max(im.flatten())
    im = (im - mini) / (maxi - mini)
    im[im < 0] = 0
    im[im > 1] = 1
    return im


def render_pyramid(pyr, levels):
    """
    prepare pyramid for displaying
    :param pyr: pyramid of an image
    :param levels: num pf levels of pyramid
    :return: rendered pyramid
    """
    res = stretch(pyr[0])
    for i in range(1, levels):
        temp = stretch(pyr[i])
        new_im = np.zeros((res.shape[0], res.shape[1] + temp.shape[1]))
        new_im[0:res.shape[0], 0:res.shape[1]] = res
        new_im[0:pyr[i].shape[0], res.shape[1]:res.shape[1] + pyr[i].shape[1]] = temp
        res = new_im
    return res


def display_pyramid(pyr, levels):
    """
    display pyramid
    :param pyr: pyramid of an image
    :param levels: num pf levels of pyramid
    :return: None
    """
    res = render_pyramid(pyr, levels)
    plt.figure()
    plt.imshow(res, cmap="gray")


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """
    :param im1: are two input grayscale images to be blended.
    :param im2: are two input grayscale images to be blended.
    :param mask: is a boolean (i.e. dtype == np.bool) mask containing True and False representing which parts
            of im1 and im2 should appear in the resulting im_blend. Note that a value of True corresponds to 1,
            and False corresponds to 0.
    :param max_levels: is the max_levels parameter you should use when generating the Gaussian and Laplacian pyramids.
    :param filter_size_im:  is the size of the Gaussian filter (an odd scalar that represents a squared filter) which
            defining the filter used in the construction of the Laplacian pyramids of im1 and im2.
    :param filter_size_mask: is the size of the Gaussian filter(an odd scalar that represents a squared filter) which
        defining the filter used in the construction of the Gaussian pyramid of mask.
    :return: blended image from the 2 images.
    """
    L1, filter_vec = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    L2 = build_laplacian_pyramid(im2, max_levels, filter_size_im)[0]
    mask_pyr = build_gaussian_pyramid(mask.astype(np.float64), max_levels, filter_size_mask)[0]
    Lout = []
    for i in range(max_levels):
        Lout.append(L1[i]*mask_pyr[i] + (1-mask_pyr[i]) * L2[i])
    coeff = [1] * max_levels
    return laplacian_to_image(Lout, filter_vec, coeff)


def blending_example1():
    im1 = imread("sun_drawing.jpg")
    im2 = imread("s.jpg")
    mask = imread("mask1.jpg", as_grey=True)
    mask = mask > 0.5
    im_blend = copy.deepcopy(im1)
    for i in range(3):
        im_blend[:, :, i] = pyramid_blending(im1[:,:,i], im2[:,:,i], mask, 3, 3, 5)
    show_4_plots(im1, im2, mask, im_blend)
    return im1, im2, mask, im_blend


def blending_example2():
    im1 = imread("Lecture.jpg")
    im2 = imread("teletubbies.jpg")
    mask = imread("mask2.jpg", as_gray=True)
    mask = mask > 0.5
    im_blend = copy.deepcopy(im1)
    for i in range(3):
        im_blend[:, :, i] = pyramid_blending(im1[:,:,i], im2[:,:,i], mask, 3, 3, 5)
    show_4_plots(im1, im2, mask, im_blend)
    return im1, im2, mask, im_blend


def show_4_plots(im1, im2, mask, im_blend):
    plt.subplots(nrows=2, ncols=2)
    plt.subplot(2, 2, 1)
    plt.imshow(im1)
    plt.subplot(2, 2, 2)
    plt.imshow(im2)
    plt.subplot(2, 2, 3)
    plt.imshow(mask, cmap="gray")
    plt.subplot(2, 2, 4)
    plt.imshow(im_blend)


if __name__ == "__main__":
    blending_example1()
    blending_example2()
