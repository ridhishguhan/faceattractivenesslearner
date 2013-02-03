import Image
import math
import matplotlib
import matplotlib.pyplot as pyplot
import numpy as np
import numpy.linalg as linalg

def float2int8(A):
    # function im = float2int8(A)
    # convert an float array image into grayscale image with contrast from 0-255
    amin = np.amin(A)
    amax = np.amax(A)
    im = ((A - amin) / (amax - amin)) * 255
    im = np.trunc(im)
    return im.astype(np.int8)

def normalize(mat):
    for i in range(mat.shape[0]):
        A = mat[i,:]
        amin = np.amin(A)
        amax = np.amax(A)
        mat[i,:] = ((A - amin) / (amax - amin))
    return mat

def ComputeNorm(x):
    # function r=ComputeNorm(x)
    # computes vector norms of x
    # x: d x m matrix, each column a vector
    # r: 1 x m matrix, each the corresponding norm (L2)
    [row, col] = x.shape
    r = np.zeros((1,col))
    for i in range(col):
        r[0,i] = linalg.norm(x[:,i])
    return r

def norm(x):
    x = x.ravel()
    return np.sqrt(np.dot(x.T,x))

# compute accuracy
def accuracy(A):
    print "Trace : ", np.trace(A), " Sum : ",  np.sum(A.reshape(-1))
    return np.trace(A)/(0.0+np.sum(A.reshape(-1)))

def approxToDiagonal(A, approx = True):
    [r,c] = A.shape
    i = 0
    mask_mat = np.identity(r)
    while i < r:
        mask_vector = mask_mat[i]
        dotp = np.array(A[i,:]) * np.array(mask_vector)
        A[i,:] = dotp
        if approx and A[i,i] < 0.7:
            A[i,i] = 0.7
        i += 1
    return A

def getCovarianceMatrix(A):
    [r,c] = A.shape
    S = np.cov(A, rowvar = 1, bias = 1)
    print "Covar Shape : ", S.shape
    return S

def getImageFromArray(arr, shape):
    #print arr.shape
    mat = arr.reshape(shape).copy()
    mat = float2int8(mat)
    image = Image.fromarray(mat,"L")
    return image

#shows a particular image
def showFeature(arr, save = False, path = ""):
    image = getImageFromArray(arr)
    if save:
        image.save(path)
    else:
        image.show()

#for reconstructing Image from PCA space
def reconstruct(Y, pca, mean):
    figure = pyplot.figure()

    # adding gestures to figure
    for i in range(10):
        print i
        y = Y[:,10 * i]
        x = np.dot(pca,y) + mean
        image = getImageFromArray(x)
        image = image.rotate(180)
        pic = figure.add_subplot(5,2,i+1)
        pic.xaxis.set_visible(False)
        pic.yaxis.set_visible(False)
        pic.set_title("Recon Hand : " + str(i+1))
        pic.imshow(image, matplotlib.cm.get_cmap('gist_gray'))
    figure.show()
    return

def packFaceListAsMatrix(list):
    A = []
    for face in list:
        imarr = face.face_array
        A.append(imarr)
    A = np.array(A)
    return np.matrix(A.T)

def packListAsMatrix(list):
    A = []
    for item in list:
        A.append(item)
    A = np.array(A)
    return np.matrix(A.T)

def getAsList(data):
    A = []
    for i in range(data.shape[1]):
        x = data[:,i].T
        x = np.array(np.reshape(x, -1))
        A.append(x)
    return A

# Given classification, actual indices and labels
# return confusion matrix and accuracy
def getConfusionMatrixAndAccuracy(result, test_data):
    conf = np.matrix(np.zeros([3,3]))
    for i in range(len(result)):
        ai = test_data[i].attractiveness
        conf[ai,result[i]] += 1
    return conf, accuracy(conf)

#print classification on screen
# also returns conf matrix and accuracy
def printClassification(result, test_data, print_stat = True):
    if print_stat:
        print "Classification : "
        misclas = 0
        for i in range(len(test_data)):
            print "Test : ", i + 1," Classify : ", result[i], " Actual : ", test_data[i].attractiveness
            if result[i] != test_data[i].attractiveness:
                misclas += 1
        print "Misclassifications : ", misclas
    [conf, accu] = getConfusionMatrixAndAccuracy(result, test_data)
    if print_stat:
        print "Accuracy : ",accu
    return conf, accu

def logsumexp(arr, axis=0):
    """Computes the sum of arr assuming arr is in the log domain.

    Returns log(sum(exp(arr))) while minimizing the possibility of
    over/underflow.

    Examples
    --------

    >>> import numpy as np
    >>> from sklearn.utils.extmath import logsumexp
    >>> a = np.arange(10)
    >>> np.log(np.sum(np.exp(a)))
    9.4586297444267107
    >>> logsumexp(a)
    9.4586297444267107
    """
    arr = np.rollaxis(arr, axis)
    # Use the max to normalize, as with the log this is what accumulates
    # the less errors
    vmax = arr.max(axis=0)
    out = np.log(np.sum(np.exp(arr - vmax), axis=0))
    out += vmax
    return out