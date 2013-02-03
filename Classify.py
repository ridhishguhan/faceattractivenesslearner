import numpy as np
import Utils

class Classifier:
    training = None
    train_arr = None
    classes = None

    def __init__(self, training, train_arr, CLASSES = 3):
        self.training = training
        self.train_arr = train_arr
        self.classes = CLASSES

    #KNN Classification method
    def OneNNClassify(self, test_set, K):
        # KNN Method
        # for each test sample t
        #    for each training sample tr
        #        compute norm |t - tr|
        #    choose top norm
        #    class which it belongs to is classification
        [tr,tc] = test_set.shape
        [trr,trc] = self.train_arr.shape
        result = np.array(np.zeros([tc]))
        i = 0
        #print "KNN : with K = ",K
        while i < tc:
            x = test_set[:,i]
            xmat = np.tile(x,(1,trc))
            xmat = xmat - self.train_arr
            norms = Utils.ComputeNorm(xmat)
            closest_train = np.argmin(norms)
            which_train = self.training[closest_train]
            attr = which_train.attractiveness
            result[i] = attr
            #print "Class : ",result[i]
            i += 1
        return result