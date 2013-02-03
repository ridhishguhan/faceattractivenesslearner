import numpy as np
import numpy.linalg as la
import numpy.random as npr
import random as pr

import Utils
npa = np.array

#eps = 10**-9
eps = np.finfo(float).eps
class Normal(object):

    def __init__(self, dim, mu = None, sigma = None):
        self.dim = dim # full data dimension
        self.update(npa(mu,dtype='float64'),npa(sigma,dtype='float64'))

    def update(self, mu, sigma):
        self.mu = mu
        self.E = sigma

        det = None
        if self.dim == 1:
            self.A = 1.0 / self.E
            det = np.fabs(self.E[0])
        else:
            # precision matrix
            det = np.fabs(la.det(self.E))
            #if det == 0:
            #    self.E = Utils.approxToDiagonal(self.E, approx = False)
            #    det = np.fabs(la.det(self.E))
            self.A = la.inv(self.E)
            det = np.fabs(la.det(self.E))

        self.factor = ((2.0 * np.pi)**(self.dim / 2.0)) * (det)**(0.5)
        print "Updated to : ", self.factor

    def __str__(self):
        return "%s\n%s" % (str(self.mu), str(self.E))

    def mean(self):
        return self.mu

    def covariance(self):
        return self.E

    def pdf(self, x):
        np.float64(x)
        dx = x - self.mu
        A = self.A
        fE = self.factor
        #print "PDF | Factor : ",fE
        dotp = -0.5 * np.dot(np.dot(dx.T,A),dx)
        #if np.fabs(dotp) == 0:
        #    dotp = eps
        numer = np.exp(dotp)# + eps
        #print "PDF | Numer : ",numer
        prob =  numer / fE
        print "PDF | Prob : ",prob
        if np.isnan(fE) or np.isnan(numer) or np.isnan(prob):
            raise ValueError("Error in PDF function")
        return prob