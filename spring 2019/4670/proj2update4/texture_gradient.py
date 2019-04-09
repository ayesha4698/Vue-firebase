import numpy as np
import math
import cv2

import scipy 
from scipy.signal import convolve2d as convolve
import matplotlib.pyplot as plt

#########################################################
# Part A: Image Processing Functions
#########################################################

# TODO:PA2 Fill in this function


def normalizeImage(cvImage, minIn, maxIn, minOut, maxOut):
    '''
    Take image and map its values linearly from [min_in, max_in]
    to [min_out, max_out]. Assume the image does not contain
    values outside of the [min_in, max_in] range.

    Parameters:
            cvImage - a (m x n) or (m x n x 3) image.
            minIn - the minimum of the input range
            maxIn - the maximum of the input range
            minOut - the minimum of the output range
            maxOut - the maximum of the output range

    Return:
            renormalized - the linearly rescaled version of cvImage.
    '''
    in_range = maxIn - minIn
    out_range = maxOut - minOut

    if cvImage.ndim == 3:
        (height, width, channel) = cvImage.shape
    else:
        (height, width) = cvImage.shape

    if is_rgb(cvImage):
        renormalized = np.zeros((height, width, 3))
        for img_row in range(height):
            for img_col in range(width):
                renormalized[img_row][img_col] = \
                    normal(cvImage[img_row][img_col], minIn, minOut, in_range, out_range, True)
    else:
        renormalized = np.zeros((height, width))
        for img_row in range(height):
            for img_col in range(width):
                renormalized[img_row][img_col] = \
                    normal(cvImage[img_row][img_col], minIn, minOut, in_range, out_range, False)

    return renormalized


def normal(x, minIn, minOut, in_range, out_range, is_rgb):
    if is_rgb:
        red = ((x[0] - minIn) / in_range) * out_range + minOut
        green = ((x[1] - minIn) / in_range) * out_range + minOut
        blue = ((x[2] - minIn) / in_range) * out_range + minOut
        return [red, green, blue]
    else:
        return ((x - minIn) / in_range) * out_range + minOut


    # delta = (maxOut - minOut) / (maxIn - minIn)
    # for i in range(0, height):
    #     for j in range(0, width):
    #         if cvImage.ndim == 3:
    #             for k in range(0, 3):
    #                 cvImage[i, j, k] = minOut + \
    #                     (cvImage[i, j, k] - minIn) * delta
    #         else:
    #             cvImage[i, j] = minOut + (cvImage[i, j] - minIn) * delta
    #
    # return cvImage

#    max_array = abs(np.amax(gradientImage))
#    min_array = abs(np.amin(gradientImage))
#    max_val = max(max_array, min_array)
#
#    return normalizeImage(gradientImage, -max_val, max_val, 0, 255).astype(np.uint8)

def getDisplayGradient(gradientImage):
    '''
    Use the normalizeImage function to map a 2d gradient with one
    or more channels such that where the gradient is zero, the image
    has 50% percent brightness. Brightness should be a linear function
    of the input pixel value. You should not clamp, and
    the output pixels should not fall outside of the range of the uint8
    datatype.

    Parameters:
            gradientImage - a per-pixel or per-pixel-channel gradient array
                                            either (m x n) or (m x n x 3). May have any
                                            numerical datatype.

    Return:
            displayGrad - a rescaled image array with a uint8 datatype.
    '''
    max_arr = abs(np.max(gradientImage))
    min_arr = abs(np.min(gradientImage))
    max_val = max(max_arr, min_arr)
    return normalizeImage(gradientImage, -max_val, max_val, 0, 255).astype(np.uint8)


def takeXGradient(cvImage):
    '''
    Compute the x-derivative of the input image with an appropriate
    Sobel implementation. Should return an array made of floating
    point numbers.

    Parameters:
            cvImage - an (m x n) or (m x n x 3) image

    Return:
            xGrad - the derivative of cvImage at each position w.r.t. the x axis.

    '''

    x1 = np.array([[1, 0, -1]])
    x2 = np.array([[1], [2], [1]])

    if is_rgb(cvImage):
        red = np.zeros((cvImage.shape[0], cvImage.shape[1]))
        green = np.zeros((cvImage.shape[0], cvImage.shape[1]))
        blue = np.zeros((cvImage.shape[0], cvImage.shape[1]))
        for img_row in range(cvImage.shape[0]):
            for img_col in range(cvImage.shape[1]):
                red[img_row][img_col] = cvImage[img_row][img_col][0]
                green[img_row][img_col] = cvImage[img_row][img_col][1]
                blue[img_row][img_col] = cvImage[img_row][img_col][2]

        red_grad = convolve(convolve(red, x1, mode='same'), x2, mode='same')
        green_grad = convolve(convolve(green, x1, mode='same'), x2, mode='same')
        blue_grad = convolve(convolve(blue, x1, mode='same'), x2, mode='same')

        xGrad = np.zeros((cvImage.shape[0], cvImage.shape[1], 3))
        for img_row in range(cvImage.shape[0]):
            for img_col in range(cvImage.shape[1]):
                xGrad[img_row][img_col] = \
                    [red_grad[img_row][img_col], green_grad[img_row][img_col], blue_grad[img_row][img_col]]

    else:
        xGrad = convolve(convolve(cvImage, x1, mode='same'), x2, mode='same')

    return xGrad


def takeYGradient(cvImage):
    '''
    Compute the y-derivative of the input image with an appropriate
    Sobel implementation. Should return an array made of floating
    point numbers.

    Parameters:
            cvImage - an (m x n) or (m x n x 3) image

    Return:
            yGrad - the derivative of cvImage at each position w.r.t. the y axis.
    '''

    # filter_kernel = [[1, 2, 1],
    #                  [0, 0, 0],
    #                  [-1, -2, -1]]
    #
    # if cvImage.ndim == 3:
    #     (height, width, channel) = cvImage.shape
    #     for k in range(0, 3):
    #         cvImage[:, :, k] = convolve(
    #             cvImage[:, :, k], filter_kernel, mode="same", boundary="fill")
    #     return cvImage
    #
    # else:
    #     img = convolve(cvImage, filter_kernel, mode="same", boundary="fill")
    #     return img
    y1 = np.array([[1, 2, 1]])
    y2 = np.array([[1], [0], [-1]])

    if is_rgb(cvImage):
        red = np.zeros((cvImage.shape[0], cvImage.shape[1]))
        green = np.zeros((cvImage.shape[0], cvImage.shape[1]))
        blue = np.zeros((cvImage.shape[0], cvImage.shape[1]))
        for img_row in range(cvImage.shape[0]):
            for img_col in range(cvImage.shape[1]):
                red[img_row][img_col] = cvImage[img_row][img_col][0]
                green[img_row][img_col] = cvImage[img_row][img_col][1]
                blue[img_row][img_col] = cvImage[img_row][img_col][2]

        red_grad = convolve(convolve(red, y1, mode='same'), y2, mode='same')
        green_grad = convolve(convolve(green, y1, mode='same'), y2, mode='same')
        blue_grad = convolve(convolve(blue, y1, mode='same'), y2, mode='same')

        yGrad = np.zeros((cvImage.shape[0], cvImage.shape[1], 3))
        for img_row in range(cvImage.shape[0]):
            for img_col in range(cvImage.shape[1]):
                yGrad[img_row][img_col] = \
                    [red_grad[img_row][img_col], green_grad[img_row][img_col], blue_grad[img_row][img_col]]

    else:
        yGrad = convolve(convolve(cvImage, y1, mode='same'), y2, mode='same')

    return yGrad

def is_rgb(img):
    return img.ndim == 3


def takeGradientMag(cvImage):
    '''
    Compute the gradient magnitude of the input image for each
    pixel in the image.

    Parameters:
            cvImage - an (m x n) or (m x n x 3) image

    Return:
            gradMag - the magnitude of the 2D gradient of cvImage.
                              if multiple channels, handle each channel seperately.
    '''

    # threeDim = cvImage.ndim == 3
    # if threeDim:
    #     (height, width, channel) = cvImage.shape
    # else:
    #     (height, width) = cvImage.shape

    # xGradient = np.power(takeXGradient(cvImage), 2)
    # yGradient = np.power(takeYGradient(cvImage), 2)
    #
    # return np.sqrt(xGradient + yGradient)
    xGradient = takeXGradient(cvImage)
    yGradient = takeYGradient(cvImage)
    if is_rgb(cvImage):
        new_img = np.zeros((cvImage.shape[0], cvImage.shape[1], 3))
        for img_row in range(cvImage.shape[0]):
            for img_col in range(cvImage.shape[1]):
                red_sqr_x = xGradient[img_row][img_col][0] * xGradient[img_row][img_col][0]
                green_sqr_x = xGradient[img_row][img_col][1] * xGradient[img_row][img_col][1]
                blue_sqr_x = xGradient[img_row][img_col][2] * xGradient[img_row][img_col][2]

                red_sqr_y = yGradient[img_row][img_col][0] * yGradient[img_row][img_col][0]
                green_sqr_y = yGradient[img_row][img_col][1] * yGradient[img_row][img_col][1]
                blue_sqr_y = yGradient[img_row][img_col][2] * yGradient[img_row][img_col][2]

                red, green, blue = math.sqrt(red_sqr_x + red_sqr_y), math.sqrt(green_sqr_x + green_sqr_y), math.sqrt(
                    blue_sqr_x + blue_sqr_y)
                new_img[img_row][img_col] = [red, green, blue]
    else:
        new_img = np.zeros((cvImage.shape[0], cvImage.shape[1]))
        for img_row in range(cvImage.shape[0]):
            for img_col in range(cvImage.shape[1]):
                sqr_x = xGradient[img_row][img_col] * xGradient[img_row][img_col]
                sqr_y = yGradient[img_row][img_col] * yGradient[img_row][img_col]
                new_img[img_row][img_col] = math.sqrt(sqr_x + sqr_y)
    return new_img


#########################################################
# Part B: k-Means Segmentation Functions
#########################################################

# TODO:PA2 Fill in this function
def computeDistance(X, C):
    '''
    Compute each pairwise squared distance between elements
    of X and C using Euclidean metric. Do not take squared root

    Parameters:
            X - feature matrix (n x d)
            C - centroid matrix (m x d)
    Returns:
            dist - distance matrix m x n)
    '''
    (m, d2) = C.shape
    (n, d1) = X.shape

    x_squared = X ** 2
    c_squared = C ** 2

    sumX = np.sum(x_squared, axis=1)
    sumC = np.sum(c_squared, axis=1)

    term = -2 * np.dot(C, np.transpose(X))

    sumC = np.tile(sumC, (n, 1))
    sumX = np.tile(sumX, (m, 1))

    result = sumC.transpose() + sumX + term
    return result


def getFeats(img):
    '''
    Compute and return feature vectors for each pixel in the image.

    Normalize the x and y coordinates to the range [0, 1).
    Normalize each RGB value in the image by the maximum value in any channel.

    Note: do not use nested for loops to iterate through the image!
    Note: origin (0, 0) is top left

    Parameters:
            img - Input image (m x n x 3)

    Returns:
            feats - Array of feature vectors (m x n x 5)


    '''
###################################################################################
    (m, n, channel) = img.shape

    max_val = abs(np.max((img)))
    img = np.divide(img, max_val)

    grid = np.indices((m, n))

    normalizedX = grid[0]/(m-1)  # m
    normalizedX = np.reshape(normalizedX, (m,n,1))

    normalizedY = grid[1]/(n-1)  # n
    normalizedY = np.reshape(normalizedY, (m, n, 1))

    img = np.concatenate((img, normalizedY), axis = 2 )
    img = np.concatenate((img, normalizedX), axis = 2 )

    return img

def runKmeans(feats, k, max_iter=100):
    '''

    Parameters:
            feats - array of points to cluster (n x d)
            k - number of clusters to create
            max_iter - number of iterations of the algorithm to run

    Returns:
            labels - array of labels (n)
    '''
    np.random.seed(11)
    idx = np.random.choice(feats.shape[0], k)
    centroids = feats[idx, :]
    for i in range(max_iter):
        distances = computeDistance(feats, centroids)
        mindist = np.min(distances, axis=0)
        labels = np.argmin(distances, axis=0)
        for j in range(k):
            if np.sum(labels == j) > 0:
                centroids[j, :] = np.mean(feats[labels == j, :], axis=0)
            else:
                idx = np.random.choice(feats.shape[0])
                labels[idx] = j
                centroids[j, :] = feats[idx, :]
    return labels


def segmentKmeans(img, num):
    '''
    Execute a color-based k-means segmentation
    '''
    feats = getFeats(img)
    segments = runKmeans(feats.reshape((-1, feats.shape[2])), num, 100)
    return segments.reshape((img.shape[0], img.shape[1]))


#########################################################
# Part C: Texture Based Edge Detection
#########################################################
# TODO:PA2 Fill in this function
def getOrientedFilter(f, theta, size=11):
    '''
            Return a oriented filter of size `size` oriented at angle theta.

            Parameters:
                    f - function that takes in vectors of x, y coordinates
                            and returns a gaussian filter
                    theta - angle (radians) needed to generate x and y
                    size - size of the filter

            Returns:
                    gaussFilter - a (size, size) array

    '''
    grid = np.indices((size, size))
    offset = (size-1)/2
    xGrid = grid[0].flatten() - offset
    yGrid = grid[1].flatten() - offset

    rotateX = (yGrid * np.cos(theta)) - (xGrid * np.sin(theta))
    rotateY = (xGrid * np.cos(theta)) + (yGrid * np.sin(theta))

    matrix = f(rotateX, rotateY)
    matrix = np.reshape(matrix, (size, size))

    return matrix

# TODO:PA2 Fill in this function


def getDoG(x, y, sigma_1=0.7, sigma_2=2):
    '''
            Return a Difference of Gaussians filter value for each element in zip(x, y).

            Parameters:
                    x - array of x coordinates (n)
                    y - array of y coordinates (n)
                    sigma_1 - standard deviation of the first filter
                    sigma_2 - standard deviation of the second filter
            Returns:
                    diff - array of filter values (n)
    '''
    n = x.shape[0]
    val1 = np.zeros(n)
    val2 = np.zeros(n)
    for i in range(0, n):
        val1[i] = getGaussian(x[i], y[i], sigma_1)
        val2[i] = getGaussian(x[i], y[i], sigma_2)
    return np.subtract(val1, val2)


def getGaussDeriv(x, y, sigma_x=0.7, sigma_y=2):
    return -x*np.exp(-(x*x/(2*sigma_x*sigma_x) + y*y/(2*sigma_y*sigma_y)))


def getGauss2Deriv(x, y, sigma_x=0.7, sigma_y=2):
    ygauss = np.exp(-y*y/(2*sigma_y*sigma_y))
    xgauss = np.exp(-x*x/(2*sigma_x*sigma_x))
    val = xgauss*ygauss*(x*x/(sigma_x**4) - 1/(sigma_x*sigma_x))
    return val


def getGaussian(x, y, sigma=1):
    gauss = np.exp(-(x*x+y*y)/(2*sigma*sigma))/(2*np.pi*sigma*sigma)
    return gauss


def getFilterBank():
    filters = []
    for theta in np.arange(0, np.pi, np.pi/8):
        filters.append(getOrientedFilter(getGaussDeriv, theta))
        filters.append(getOrientedFilter(getGauss2Deriv, theta))
    filters.append(getOrientedFilter(getDoG, 0))
    return filters


# TODO:PA2 Fill in this function
def getTextons(img, vocab_size=100):
    '''
            Compute textons and assign each pixel to a texton.

            Preprocess each channel by normalizing (divide by 255) and convolving with gaussian.

            Each pixel is associated with a feature vector of 3*len(filter_bank) containing
            one entry for each channel for each filter such that
            [R's filters, G's filters, B's filters]. Note, this is not the feature vector
            returned by getFeats in Part B.

            These vectors are then clustered using K-means and converted into one-hot labels
            for each pixel.

            Parameters:
                    img - input image (n x m x 3)
                    vocab_size - texton dimension
            Returns:
                    labels - one hot labels for each pixel in the image (n x m x vocab_size)
                    features - numpy array of features (n*m, 3 * len(filter_bank))
    '''

    gaussian = getOrientedFilter(getGaussian, 0)

    filter_bank = getFilterBank()

    # TODO:PA2 Fill in this part
    (m, n, c) = img.shape
    feats = np.zeros((m, n, 1))

    red = np.zeros((m, n))
    green = np.zeros((m, n))
    blue = np.zeros((m, n))

    # Creates mxn array of each channel
    for r in range(0, m):
        for c in range(0, n):
            red[r][c] = img[r][c][0]
            green[r][c] = img[r][c][1]
            blue[r][c] = img[r][c][2]

    # Preprocess
    red = convolve((red/255), gaussian, mode="same")
    green = convolve((green/255), gaussian, mode="same")
    blue = convolve((blue/255), gaussian, mode="same")

    r_filtered = []
    g_filtered = []
    b_filtered = []

    # Go through all the filters
    for f in range(0, len(filter_bank)):
        r_filtered.append(convolve(red, filter_bank[f], mode="same")[
                          :, :, np.newaxis])
        g_filtered.append(convolve(green, filter_bank[f], mode="same")[
                          :, :, np.newaxis])
        b_filtered.append(convolve(blue, filter_bank[f], mode="same")[
                          :, :, np.newaxis])

    l = r_filtered + g_filtered + b_filtered
    feats = np.concatenate(l, axis=2)
    feats = feats.reshape(n*m, feats.shape[2])

    # END: PA2

    labels = runKmeans(feats, vocab_size, 500)
    onehotlabels = np.zeros((labels.shape[0], vocab_size))
    for i in range(labels.shape[0]):
        onehotlabels[i, labels[i]] = 1

    return onehotlabels.reshape((img.shape[0], img.shape[1], -1)), feats


# TODO:PA2 Fill in this function
def getMasks(theta, size):
    '''
            Create two filters (size x size) to partition an image into two halves
            based on orientation (theta).

            For theta = pi/4 and size = 5, the first filter returned would be:
            array([[False, False, False, False, False],
                       [ True, False, False, False, False],
                       [ True,  True, False, False, False],
                       [ True,  True,  True, False, False],
                       [ True,  True,  True,  True, False]], dtype=bool)

            Parameters:
                    theta - the angle of the line across which to divide
                    size - the size of the filter to produce
            Returns:
                    mask_1 - the filter with True values for the left side of the image
                    mask_2 - the filter with True values for the right side of the image


    '''

    #if even do this
    theta2 = theta - math.pi
    midd = (size ) // 2
    if(size % 2 == 0):
        x_grid = (np.tile(np.arange(-midd + 1, midd + 1), (size, 1)) -.5 ) * -1

        y_o = np.zeros((size, 1))
        for i in range(-midd + 1, midd + 1):
            y_o [i - midd -1] = -i +.5

        y_grid = np.repeat(y_o, size, axis=1)
        yx = np.arctan2(y_grid, x_grid)

        left = ((yx > 0) & (yx < theta)) | ((yx <= 0) & (yx > theta2))
        right = ((yx > 0) & (yx > theta)) | ((yx <= 0) & (yx < theta2))
        if (theta == 0):
            for i in range(0, size):
                left[midd][i] = False
                right[midd][i] = False


        return left, right

    else:
        x_grid = np.tile(np.arange(-midd, midd + 1), (size,1)) * -1;
        y_o = np.zeros((size, 1))
        for i in range(-midd, midd + 1):
            y_o[i - midd - 1] = -i
        y_grid = np.repeat(y_o, size, axis = 1)
        yx = np.arctan2(y_grid, x_grid)



        left = ((yx > 0 ) & (yx <theta )) | ((yx<= 0 ) & (yx > theta2))
        right = ((yx > 0 ) & (yx > theta )) | ((yx<= 0 ) & (yx < theta2))
        left[midd][midd] = False
        right[midd][midd] = False
        if(theta == 0):
            for i in range(0, size):
                left[midd][i] = False
                right[midd][i] = False

    return left, right




# TODO:PA2 Fill in this function


def computeTextureGradient(img, vocab_size=100, r=10):
    '''
            Use getTextons to get the one-hot labels for img and flatten to a
            1D array of pixels' one-hot vectors.

            Generate masks for thetas from [0, pi] with step size pi/4.

            Compute differences of histograms in oriented half-discs of width r.

            Parameters:
                    img - (m x n x 3) image
                    vocab_size - how many textons to split into
                    r - width of oriented half-discs
            Returns:
                    G - (n x m x 4) where 4 is len(theta)

    '''

    (Labels, features) = np.array(getTextons(img, vocab_size))
    (m, n, z) = img.shape
    G = np.zeros((m, n, 4))
    masks_lst = np.zeros((4), dtype=bool)

    #generate masks
    mask0L, mask0R =  getMasks((0), (2*r + 1))
    mask1L, mask1R = getMasks((math.pi /4), (2 * r + 1))
    mask2L, mask2R = getMasks((math.pi / 2), (2 * r + 1))
    mask3L, mask3R = getMasks((3* math.pi/ 4), (2 * r + 1))

    hot_labels = np.zeros((m, n))



    for x in range(0, m):
        for y in range(0, n):
            hot_labels[x][y] = (np.where(Labels[x][y] == 1))[0][0] + 1


    hot_labels = hot_labels.astype(int);


    for i in range(r, m - r):
        for j in range(r, n - r):
            patch = hot_labels[i - r:i + r + 1, j - r:j + r + 1]
            f_lst = []
            f1 = np.bincount((np.multiply(mask0L, patch)).flatten(), minlength=vocab_size + 1)[1:]
            f_lst.append(f1)
            if np.sum(f1) != 0:
                f1 = f1 / np.sum(f1)

            f2 = np.bincount((np.multiply(mask0R, patch)).flatten(), minlength=vocab_size + 1)[1:]
            f_lst.append(f2)
            if np.sum(f2) != 0:
                f2 = f2 / np.sum(f2)

            f3 = np.bincount((np.multiply(mask1L, patch)).flatten(), minlength=vocab_size + 1)[1:]
            f_lst.append(f3)
            if np.sum(f3) != 0:
                f3 = f3 / np.sum(f3)

            f4 = np.bincount((np.multiply(mask1R, patch)).flatten(), minlength=vocab_size + 1)[1:]
            f_lst.append(f4)
            if np.sum(f4) != 0:
                f4 = f4 / np.sum(f4)

            f5 = np.bincount((np.multiply(mask2L, patch)).flatten(), minlength=vocab_size + 1)[1:]
            f_lst.append(f5)
            if np.sum(f5) != 0:
                f5 = f5 / np.sum(f5)

            f6 = np.bincount((np.multiply(mask2R, patch)).flatten(), minlength=vocab_size + 1)[1:]
            f_lst.append(f6)
            if np.sum(f6) != 0:
                f6 = f6 / np.sum(f6)

            f7 = np.bincount((np.multiply(mask3L, patch)).flatten(), minlength=vocab_size + 1)[1:]
            f_lst.append(f7)
            if np.sum(f7) != 0:
                f7 = f7 / np.sum(f7)

            f8 = np.bincount((np.multiply(mask3R, patch)).flatten(), minlength=vocab_size + 1)[1:]
            f_lst.append(f8)
            if np.sum(f8) != 0:
                f8 = f8 / np.sum(f8)






            t0 = computeDistance(f1.reshape((1, vocab_size)), f2.reshape((1, vocab_size)))
            t1 = computeDistance(f3.reshape((1, vocab_size)), f4.reshape((1, vocab_size)))
            t2 = computeDistance(f5.reshape((1, vocab_size)), f6.reshape((1, vocab_size)))
            t3 = computeDistance(f7.reshape((1, vocab_size)), f8.reshape((1, vocab_size)))
            G[i][j] = [t0, t1, t2, t3]

    return G.astype(np.float64)






