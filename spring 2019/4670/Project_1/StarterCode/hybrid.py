import cv2
import numpy as np

def cross_correlation_2d(img, kernel):
    '''Given a kernel of arbitrary m x n dimensions, with both m and n being
    odd, compute the cross correlation of the given image with the given
    kernel, such that the output is of the same dimensions as the image and that
    you assume the pixels out of the bounds of the image to be zero. Note that
    you need to apply the kernel to each channel separately, if the given image
    is an RGB image.
    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).
    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN

    # Image Width & Height & Depth

    # Initialize Returned Image
    if img.ndim == 3:
        (height, width, channel) = img.shape
        output = np.zeros((height, width, 3))
    else:
        (height, width) = img.shape
        output = np.zeros((height, width))

    # Kernel Width & Height
    (kHeight, kWidth) = kernel.shape

    # Kernel K Value
    kh = (kHeight - 1) // 2
    kw = (kWidth - 1) // 2

    for i in range(0, height):
        for j in range(0, width):
            Gij = 0
            for u in range(-kh, kh + 1):
                for v in range(-kw, kw + 1):
                    a = i + u
                    b = j + v

                    # Check for Out of Bounds
                    outofBounds = a < 0 or a >= height or b < 0 or b >= width

                    # RGB Image vs Grayscale
                    if img.ndim == 3:
                        if not outofBounds:
                            Gij = Gij + np.dot(
                                kernel[u + kh, v + kw], np.array(
                                    (img[a, b, 0], img[a, b, 1], img[a, b, 2])))
                    else:
                        if not outofBounds:
                            Gij = Gij + (kernel[u + kh, v + kw] * img[a, b])

            output[i, j] = Gij
    return output
    # TODO-BLOCK-END


def convolve_2d(img, kernel):
    '''Use cross_correlation_2d() to carry out a 2D convolution.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    if img.ndim == 3:
        (height, width, channel) = img.shape
        output = np.zeros((height, width, 3))
    else:
        (height, width) = img.shape
        output = np.zeros((height, width))

    new_img = cross_correlation_2d(img, kernel)
    if img.ndim == 3:
        return output
    else:



    # TODO-BLOCK-END


def gaussian_blur_kernel_2d(sigma, height, width):
    '''Return a Gaussian blur kernel of the given dimensions and with the given
    sigma. Note that width and height are different.

    Input:
        sigma:  The parameter that controls the radius of the Gaussian blur.
                Note that, in our case, it is a circular Gaussian (symmetric
                across height and width).
        height: The height of the kernel.
        width:  The width of the kernel.

        ******
        It's also acceptable if your parameters are in the order of
        sigma, width, height.
        ******

    Output:
        Return a kernel of dimensions width x height such that convolving it
        with an image results in a Gaussian-blurred image.
    '''
    # TODO-BLOCK-BEGIN

    # TODO-BLOCK-END


def low_pass(img, sigma, size):
    '''Filter the image as if its filtered with a low pass filter of the given
    sigma and a square kernel of the given size. A low pass filter supresses
    the higher frequency components (finer details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    # TODO-BLOCK-END


def high_pass(img, sigma, size):
    '''Filter the image as if its filtered with a high pass filter of the given
    sigma and a square kernel of the given size. A high pass filter suppresses
    the lower frequency components (coarse details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    # TODO-BLOCK-END


def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
                        high_low2, mixin_ratio):
    '''This function adds two images to create a hybrid image, based on
    parameters specified by the user.'''
    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)

    img1 *= (1 - mixin_ratio)
    img2 *= mixin_ratio
    hybrid_img = (img1 + img2)
    if len(hybrid_img.shape) == 3:  # if its an RGB image
        for c in range(3):
            hybrid_img[:, :, c] /= np.amax(hybrid_img[:, :, c])
    else:
        hybrid_img /= np.amax(hybrid_img)
    return (hybrid_img * 255).astype(np.uint8)


