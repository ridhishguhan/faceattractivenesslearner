import numpy as np
import Utils

# some small value
_EPS = np.finfo(float).eps

class NMF:
    def __init__(self,data,basecount):
        self.data = data
        self.k = basecount
    data = None
    k = None
    W,H = None, None

    def rand_initialize(self):
        [n,m] = self.data.shape
        self.W = np.random.mtrand.rand(n,self.k)
        self.H = np.random.mtrand.rand(self.k,m)
        print "W Shape : ", self.W.shape
        print "H Shape : ", self.H.shape

    def perform(self,maxiter,tol = _EPS):
        [n,samples] = self.data.shape
        w0 = self.W
        h0 = self.H
        a = self.data
        eps = 10**-9
        err0 = 0
        count = 0
        for i in range(maxiter):
            # Multiplicative update formula
            numer = np.dot(w0.T,a);
            #print "w0.T * a : ",numer.shape

            deno = np.dot(np.dot(w0.T,w0 ),h0) + eps
            #print "(np.dot(np.dot(w0.T,w0 ),h0) : ", deno.shape

            h = np.multiply(h0,np.divide(numer,deno))

            numer = np.dot(a,h.T)
            deno = (np.dot(w0,np.dot(h,h.T)) + eps)

            #print "np.dot(a,h.T) : ",numer.shape
            #print "np.dot(w0,np.dot(h,h.T)) : ",deno.shape
            #print "w0 : ",w0.shape

            w = np.multiply(w0,np.true_divide(numer,deno))
            w0 = w
            h0 = h
            err = np.linalg.norm(a-np.dot(w,h), 'fro')

            print "Iteration : ",i+1,"/",maxiter," | Error = ",err, " Delta : ", np.abs((err - err0))

            if np.abs((err - err0)) < tol:
                count += 1
                if count > 10:
                    break
                else:
                    err0 = err
            else:
                err0 = err
                count = 0

        self.W = w0
        self.H = h0
        return w0,h0
