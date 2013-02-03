import numpy as np
import scipy.linalg as scilinalg

import Utils

class LDA:
    def __init__(self,train_data,Sw,Sb):
        self.train_data = train_data
        self.Sw = Sw
        self.Sb = Sb

    train_data = None
    Sw,Sb = None,None
    lda,eigVal = None, None

    def perform(self):
        [V,R] = scilinalg.eig(self.Sb,self.Sw, right = True)
        self.lda = R
        self.eigVal = V
        print "LDA Shape : ", self.lda.shape
        print "LDA Eig val Shape : ", self.eigVal.shape
        return

    def projectDataOntoLDA(self,reduceTo,data):
        W = self.lda[:,0:reduceTo]
        Y = np.dot(W.T,data)
        return Y

    def getFisherFaces(self,(fromidx,to),pca, pca_reduce):
        y = self.lda[:,fromidx-1 : to]
        y = pca.reconstruct(y,pca_reduce,False)
        images = []
        # adding gestures to figure
        for i in range(to - fromidx + 1):
            image = Utils.getImageFromArray(y[:,i],[80,64])
            image = image.rotate(180)
            images.append(image)
        return images

    def reduceFeatureDimension(self,reduceTo):
        return self.projectDataOntoLDA(reduceTo,self.train_data)