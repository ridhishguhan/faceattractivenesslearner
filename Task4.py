import matplotlib
from matplotlib import cm as cm
from matplotlib import pyplot
import numpy as np
import pickle
from pylab import *

import DataSet
import GMMInitModule
import PCAModule
import GMMCompute
import Utils

plot_save_path = "E:\\EE5907R\\project2\\"

def plot4a(bases):
    figure = pyplot.figure()
    for x in range(len(bases)):
        title = str(x+1)
        image = bases[x]

        sp = figure.add_subplot(3,3,x+1)
        sp.xaxis.set_visible(False)
        sp.yaxis.set_visible(False)
        sp.set_title(title)
        sp.imshow(image,cm.gist_gray)
    pyplot.autoscale()
    figure.savefig(plot_save_path + "plot4a.png")
    show()

def task4():
    whole_train = DataSet.training_data
    whole_train.extend(DataSet.test_data)

    wt_arr = Utils.packFaceListAsMatrix(whole_train)
    print "Whole Database :",wt_arr.shape
    print "Number of Samples :",wt_arr.shape[1]

    whole_pca = PCAModule.PCA(wt_arr)
    whole_pca.perform()

    print "PCs : ", len(whole_pca.eigVal)

    PCA_DIM = 50
    reduced_dataset = whole_pca.reduceFeatureDimension(PCA_DIM)
    print "Reduced Features : ",reduced_dataset.shape
    #reduced_dataset = reduced_dataset.T
    #dlist = Utils.getAsList(reduced_dataset)
    face_gmm_init = GMMInitModule.GMMInit(PCA_DIM, 8, reduced_dataset, method = "kmeans")
    #face_gmm_init.expected_version()
    comps = face_gmm_init.comps

    print "Number of Components : ", len(comps)
    means = np.matrix(np.zeros([PCA_DIM,8]))
    covars = []
    for i in range(8):
        means[:,i] = comps[i].mu
        covars.append(comps[i].E)
    print "MUS : ", means.shape
    means_recon = whole_pca.reconstruct(means, PCA_DIM)
    images = []
    # adding gestures to figure
    for i in range(8):
        image = Utils.getImageFromArray(means_recon[:,i],[80,64])
        image = image.rotate(180)
        images.append(image)
    plot4a(images)


    wt_arr = np.array(reduced_dataset.T)
    means = np.array(means.T)
    priors = np.array(face_gmm_init.priors)
    covars = np.array(covars)
    for covar in covars:
        covar = np.array(covar.T) 

    print "Training Type : ", type(wt_arr) , wt_arr.shape
    print "Mean Type : ", type(means), means.shape
    print "Covar Type : ", type(covars) , covars.shape
    print "Priors : ", type(priors) , priors.shape

    nice_gmm = GMMCompute.GMM(8, 'full', n_iter=1000, thresh = 0.000000001)
    nice_gmm.weights_ = priors
    nice_gmm._set_covars(covars)
    nice_gmm.means_ = means

    nice_gmm.fit(wt_arr)
    if nice_gmm.converged_:
        print "GMM : Converged\nPriors : ", nice_gmm.weights_,"\nMax Log Likelihood : ", \
            nice_gmm.max_log_prob
        means = np.matrix(nice_gmm.means_.T)
        print "MUS : ", means.shape
        means_recon = whole_pca.reconstruct(means, PCA_DIM)
        images = []
        # adding gestures to figure
        for i in range(8):
            image = Utils.getImageFromArray(means_recon[:,i],[80,64])
            image = image.rotate(180)
            images.append(image)
        plot4a(images)