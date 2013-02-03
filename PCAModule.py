import Image
import numpy as np
import numpy.linalg as linalg
import Utils

#PCA Class
class PCA:
    def __init__(self,data):
        self.data = np.matrix(data)
    data = None
    pca,eigVal,mean = None, None, None

    # get the minimum number of eigen values to consider to get
    # 95% variance
    def getMinEigenValues(self):
        eigsum = np.sum(self.eigVal)
        k = 0
        for x in range(len(self.eigVal)):
            buildsum = np.sum(self.eigVal[0:x])
            if buildsum/eigsum >= 0.95:
                k = x + 1
                break;
        return k

    def perform(self):
        A = self.data
        [r,c] = A.shape
        print "A Shape : ", A.shape
        m = np.mean(A, axis = 1)
        mmat = np.tile(m, (1,c))
        print "MMAT SHAPE ",mmat.shape
        A = A - mmat
        B = np.dot(np.transpose(A), A)
        [d,v] = linalg.eig(B)
        # v is in descending sorted order
        # compute eigenvectors of scatter matrix
        W = np.dot(A,v)
        Wnorm = Utils.ComputeNorm(W)
    
        W1 = np.tile(Wnorm, (r, 1))
        W2 = W / W1
        LL = d[0:-1]
        W = W2[:,0:-1]      #omit last column, which is the nullspace
        self.pca,self.eigVal,self.mean = W, LL, m
        return

    def projectDataOntoPCA(self,reduceTo,data):
        W = self.pca[:,0:reduceTo]
        [r,c] = data.shape
        data = data - np.tile(self.mean,(1,c))
        Y = np.dot(W.T,data)
        return Y

    def getEigenFaces(self,(fromidx,to)):
        y = self.pca[:,fromidx-1 : to]
        images = []
        # adding gestures to figure
        for i in range(to - fromidx + 1):
            image = Utils.getImageFromArray(y[:,i],[80,64])
            image = image.rotate(180)
            images.append(image)
        return images

    def reduceFeatureDimension(self,reduceTo):
        return self.projectDataOntoPCA(reduceTo,self.data)

    #for reconstructing Image from PCA space
    def reconstruct(self, data, reduceTo, addMean = True):
        [r,c] = data.shape
        mmat = np.tile(self.mean, (1,c))
        x = np.dot(self.pca[:,0:reduceTo],data)
        if addMean:
            x += mmat
        return x