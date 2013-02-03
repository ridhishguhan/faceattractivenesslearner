import matplotlib
from matplotlib import cm as cm
from matplotlib import pyplot
import numpy as np
import pickle
from pylab import *

import DataSet
import NMFCompute
import Utils

plot_save_path = "E:\\EE5907R\\project2\\"

def plot2a(bases):
    figure = pyplot.figure()
    for x in range(len(bases)):
        title = str(x+1)
        image = bases[x]

        sp = figure.add_subplot(5,10,x+1)
        sp.xaxis.set_visible(False)
        sp.yaxis.set_visible(False)
        sp.set_title(title)
        sp.imshow(image,cm.gist_gray)
    pyplot.autoscale()
    figure.savefig(plot_save_path + "plot2a.png")
    show()

def task2():
    whole_train = DataSet.training_data
    whole_train.extend(DataSet.test_data)

    wt_arr = Utils.packFaceListAsMatrix(whole_train)
    print "Whole Database :",wt_arr.shape
    nmf1 = NMFCompute.NMF(wt_arr,50)
    nmf1.rand_initialize()
    [W,H] = nmf1.perform(15000, 0.1)

    print "NMF W : ", nmf1.W.shape
    print "NMF H : ", nmf1.H.shape

    images = []
    for i in range(50):
        image = Utils.getImageFromArray(nmf1.W[:,i],[80,64])
        image = image.rotate(180)
        images.append(image)
    plot2a(images)