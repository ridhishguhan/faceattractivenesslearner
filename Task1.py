import matplotlib
from matplotlib import cm as cm
from matplotlib import pyplot
import numpy as np
import pickle
from pylab import *

import Utils
import DataSet
import Classify
import PCAModule

plot_save_path = "E:\\EE5907R\\project2\\"

def plot1a(male_eigen,female_eigen):
    figure = pyplot.figure()
    for x in range(20):
        if x+1 > 10:
            title = "Female " + str(x+1-10)
            image = female_eigen[x-10]
        else:
            title = "Male "+ str(x+1)
            image = male_eigen[x]

        sp = figure.add_subplot(4,5,x+1)
        sp.xaxis.set_visible(False)
        sp.yaxis.set_visible(False)
        sp.set_title(title)
        sp.imshow(image,cm.gist_gray)
    pyplot.autoscale()
    figure.savefig(plot_save_path + "plot1a.png")
    show()

def plot1b(male_acc,fem_acc,pca_values):
    figure()
    plot(pca_values,male_acc,color="red",label="Male")
    plot(pca_values,fem_acc,color="blue",label="Female")
    legend(loc='lower right')
    xlabel('Num. of PCA Features')
    ylabel('Accuracy of 1NN classifier (%)')
    title('Q1B : PCA & 1NN vs Accuracy')
    show()

def task1():
    male_train = DataSet.male_train
    female_train = DataSet.female_train

    mt_arr = Utils.packFaceListAsMatrix(male_train)
    print "Training Array :",mt_arr.shape
    ft_arr = Utils.packFaceListAsMatrix(female_train)
    print "Training Array :",ft_arr.shape

    male_pca = PCAModule.PCA(mt_arr)
    female_pca = PCAModule.PCA(ft_arr)

    male_pca.perform()
    female_pca.perform()

    print "Male Eigen : ", len(male_pca.eigVal)
    print "Female Eigen : ", len(female_pca.eigVal)

    male_eigen = male_pca.getEigenFaces((1, 10))
    female_eigen = female_pca.getEigenFaces((1, 10))

    #create plots for 1A
    plot1a(male_eigen, female_eigen)

    male_test = DataSet.male_test
    female_test = DataSet.female_test

    mtst_arr = Utils.packFaceListAsMatrix(male_test)
    print "Test Array :",mtst_arr.shape
    ftst_arr = Utils.packFaceListAsMatrix(female_test)
    print "Test Array :",ftst_arr.shape

    PCS_MAX = 50
    male_acc = []
    fem_acc = []
    pcs_vals = np.arange(1, PCS_MAX + 1)

    for dimension in range(1,PCS_MAX + 1):
        print "Number of Principal Components : ", dimension
        m_r_tra = male_pca.reduceFeatureDimension(dimension)
        f_r_tra = female_pca.reduceFeatureDimension(dimension)

        print "Reduced training data feature dimension"

        m_r_tst = male_pca.projectDataOntoPCA(dimension, mtst_arr)
        f_r_tst = female_pca.projectDataOntoPCA(dimension, ftst_arr)

        print "Reduced test data feature dimension"

        male_classify = Classify.Classifier(male_train,m_r_tra)
        female_classify = Classify.Classifier(female_train,f_r_tra)

        print "Initialized classifier"

        male_result = male_classify.OneNNClassify(m_r_tst, 1)
        female_result = female_classify.OneNNClassify(f_r_tst, 1)

        print "Classification done"

        [conf1, macc] = Utils.printClassification(male_result, male_test, False)
        [conf2, facc] = Utils.printClassification(female_result, female_test, False)

        print "Computed Confusion matrix, accuracy"
        male_acc.append(macc)
        fem_acc.append(facc)

        #np.savez("E:\\EE5907E\\project2\\save_1b_data.pk", male_acc = male_acc,fem_acc = fem_acc, pcs_vals = pcs_vals)
    plot1b(male_acc, fem_acc, pcs_vals)