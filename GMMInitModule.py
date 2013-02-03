import scipy.cluster.vq as vq
import numpy as np
import numpy.linalg as la
import numpy.random as npr
import random as pr
npa = np.array
import sys; sys.path.append('.')
from NormalDistribution import Normal
import Utils

class GMMInit(object):

    def __init__(self, dim = None, ncomps = None, data = None, method = "random"):
        self.dim = dim
        self.ncomps = ncomps
        self.comps = []
        self.converged = False
        if method is "uniform":
            # uniformly assign data points to components then estimate the parameters
            npr.shuffle(data)
            n = len(data)
            s = n / ncomps
            for i in range(ncomps):
                self.comps.append(Normal(dim, data = data[:, i * s: (i+1) * s]))
            self.priors = np.ones(ncomps, dtype = "double") / ncomps
        elif method is "random":
            # choose ncomp points from data randomly then estimate the parameters
            mus_idx = pr.sample(np.arange(data.shape[1]),ncomps)
            mus = np.matrix(np.zeros([dim,ncomps]))
            for idx in range(ncomps):
                print "MU Selected : ",mus_idx[idx]
                mus[:,idx] = data[:,mus_idx[idx]]
            clusters = [[] for i in range(ncomps)]
            for di in range(data.shape[1]):
                cdists = np.array(np.zeros([ncomps]))
                d = data[:,di]
                for mi in range(ncomps):
                    m = mus[:,mi]
                    cdists[mi] = la.norm(d - m)
                i = np.argmin(cdists)
                clusters[i].append(di)

            for i in range(ncomps):
                print mus[:,i], clusters[i]
                cluster_mat = np.matrix(np.zeros([dim,len(clusters[i])]))
                for idx in range(len(clusters[i])):
                    cluster_mat[:,idx] = data[:,clusters[i][idx]]
                ndist = Normal(dim, mu = mus[:,i], sigma = Utils.getCovarianceMatrix(cluster_mat))
                self.comps.append(ndist)
            # prior probabilities = 1/cluster_population
            # prior probabilities = 1/cluster_population
            self.priors = []
            for i in range(ncomps):
                print "Cluster size : ",len(clusters[i])
                print "Total pop : ", data.shape[1]
                print "Prior : ", np.float64(len(clusters[i]))/np.float64(data.shape[1])
                self.priors.append(np.float64(len(clusters[i]))/np.float64(data.shape[1]))
            print "Priors : ", self.priors
        elif method is "kmeans":
            # use kmeans to initialize the parameters
            shuf_data = np.array(vq.whiten(np.transpose(data)))
            #npr.shuffle(shuf_data)
            (centroids, labels) = vq.kmeans2(shuf_data, ncomps, minit="points", iter=1000)
            print "Labels : ", len(labels) , " | ", max(labels), " : ", labels
            mus = np.matrix(np.zeros([dim,ncomps]))
            clusters = [[] for i in range(ncomps)]

            for di in range(data.shape[1]):
                clusters[labels[di]].append(di)
                mus[:,labels[di]] += data[:,di]

            for i in range(ncomps):
                print mus[:,i], " Cluster size : ", len(clusters[i])
                mus[:,i] /= np.float64(len(clusters[i]), dtype = 'float64')
                #print "Centroid : ", centroids[i].shape
                #mus[:,i] = np.reshape(centroids[i],[data.shape[0],1])
                cluster_mat = np.matrix(np.zeros([dim,len(clusters[i])]))
                for idx in range(len(clusters[i])):
                    print "Data added to cluster : ", clusters[i][idx]
                    cluster_mat[:,idx] = data[:,clusters[i][idx]]
                ndist = Normal(dim, mu = mus[:,i], sigma = Utils.getCovarianceMatrix(cluster_mat))
                self.comps.append(ndist)
        
            # prior probabilities = 1/cluster_population
            #self.priors = np.array(np.ones(ncomps, dtype="double"))
            self.priors = []
            for i in range(ncomps):
                print "Cluster size : ",len(clusters[i])
                print "Total pop : ", len(labels)
                print "Prior : ", np.float64(len(clusters[i]))/np.float64(len(labels))
                self.priors.append(np.float64(len(clusters[i]))/np.float64(len(labels)))
            print "Priors : ", self.priors
