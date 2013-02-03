import matplotlib
from matplotlib import cm as cm
from matplotlib import pyplot
import numpy as np
import pickle
from pylab import *

import Utils
import DataSet
import Classify
import LDACompute
import PCAModule

plot_save_path = "E:\\EE5907R\\project2\\"

def plot3a(male_fish,female_fish):
    figure = pyplot.figure()
    for x in range(20):
        if x+1 > 10:
            title = "Female " + str(x+1-10)
            image = female_fish[x-10]
        else:
            title = "Male "+ str(x+1)
            image = male_fish[x]

        sp = figure.add_subplot(4,5,x+1)
        sp.xaxis.set_visible(False)
        sp.yaxis.set_visible(False)
        sp.set_title(title)
        sp.imshow(image,cm.gist_gray)
    pyplot.autoscale()
    figure.savefig(plot_save_path + "plot3.png")
    show()

def plot3b(male_acc,fem_acc,lda_values):
    figure()
    plot(lda_values,male_acc,color="red",label="Male")
    plot(lda_values,fem_acc,color="blue",label="Female")
    legend(loc='lower right')
    xlabel('Num. of LDA Features')
    ylabel('Accuracy of 1NN classifier (%)')
    title('Q3B : LDA & 1NN vs Accuracy')
    show()

def task3():

    # perform pca for male, female
    male_train = DataSet.male_train
    female_train = DataSet.female_train

    mt_arr = Utils.packFaceListAsMatrix(male_train)
    print "Male Training Array :",mt_arr.shape
    ft_arr = Utils.packFaceListAsMatrix(female_train)
    print "Female Training Array :",ft_arr.shape

    male_test = DataSet.male_test
    female_test = DataSet.female_test

    mtst_arr = Utils.packFaceListAsMatrix(male_test)
    print "Test Array :",mtst_arr.shape
    ftst_arr = Utils.packFaceListAsMatrix(female_test)
    print "Test Array :",ftst_arr.shape

    male_pca = PCAModule.PCA(mt_arr)
    female_pca = PCAModule.PCA(ft_arr)

    male_pca.perform()
    female_pca.perform()

    print "Male Eigen : ", len(male_pca.eigVal)
    print "Female Eigen : ", len(female_pca.eigVal)

    mt_arr_r = male_pca.reduceFeatureDimension(100)
    ft_arr_r = female_pca.reduceFeatureDimension(100)

    print "Male Reduced : ", mt_arr_r.shape
    print "Female Reduced : ", ft_arr_r.shape

    mtst_arr = male_pca.projectDataOntoPCA(100, mtst_arr)
    ftst_arr = female_pca.projectDataOntoPCA(100, ftst_arr)
    #project each class to reduced PCA space individually

    male_train_dict = DataSet.male_train_dict
    female_train_dict = DataSet.female_train_dict

    print "Male Attr : ", len(male_train_dict[2])
    print "Female Attr : ", len(female_train_dict[2])
    print "Male Marg : ", len(male_train_dict[1])
    print "Female Marg : ", len(female_train_dict[1])
    print "Male Nattr : ", len(male_train_dict[0])
    print "Female Nattr : ", len(female_train_dict[0])

    m_attr = Utils.packFaceListAsMatrix(male_train_dict[2])
    m_marg = Utils.packFaceListAsMatrix(male_train_dict[1])
    m_nattr = Utils.packFaceListAsMatrix(male_train_dict[0])

    f_attr = Utils.packFaceListAsMatrix(female_train_dict[2])
    f_marg = Utils.packFaceListAsMatrix(female_train_dict[1])
    f_nattr = Utils.packFaceListAsMatrix(female_train_dict[0])

    m_attr_r = male_pca.projectDataOntoPCA(100, m_attr)
    m_marg_r = male_pca.projectDataOntoPCA(100, m_marg)
    m_nattr_r = male_pca.projectDataOntoPCA(100, m_nattr)

    f_attr_r = female_pca.projectDataOntoPCA(100, f_attr)
    f_marg_r = female_pca.projectDataOntoPCA(100, f_marg)
    f_nattr_r = female_pca.projectDataOntoPCA(100, f_nattr)

    f_priors = np.array([len(female_train_dict[2]),len(female_train_dict[1]),len(female_train_dict[0])])
    f_priors = np.float64(f_priors) / np.float64(len(female_train)) 
    m_priors = np.array([len(male_train_dict[2]),len(male_train_dict[1]),len(male_train_dict[0])])
    m_priors = np.float64(m_priors) / np.float64(len(male_train))
    print "Priors Female : ", f_priors
    print "Priors Male : ", m_priors

    #within class covariance matrices
    Sw = Utils.getCovarianceMatrix(m_attr_r) * m_priors[0]
    Sw += Utils.getCovarianceMatrix(m_marg_r) * m_priors[1]
    Sw += Utils.getCovarianceMatrix(m_nattr_r) * m_priors[2]
    print "Male Covariance : ", Sw

    Sw1 = Utils.getCovarianceMatrix(f_attr_r) * f_priors[0]
    Sw1 += Utils.getCovarianceMatrix(f_marg_r) * f_priors[1]
    Sw1 += Utils.getCovarianceMatrix(f_nattr_r) * f_priors[2]
    print "Female Covariance : ", Sw1

    #going to compute Sb

    tot_mean = np.sum(m_attr_r, axis = 1)
    tot_mean += np.sum(m_marg_r, axis = 1)
    tot_mean += np.sum(m_nattr_r, axis = 1)
    tot_mean = np.float64(tot_mean) / np.float64(len(male_train))
    print "Male Total Mean : ", tot_mean

    tot_mean1 = np.sum(f_attr_r, axis = 1)
    tot_mean1 += np.sum(f_marg_r, axis = 1)
    tot_mean1 += np.sum(f_nattr_r, axis = 1)
    tot_mean1 = np.float64(tot_mean1)/np.float64(len(female_train))
    print "Female Total Mean : ", tot_mean1

    m_attr_mean = np.mean(np.float64(m_attr_r), axis = 1)
    m_marg_mean = np.mean(np.float64(m_marg_r), axis = 1)
    m_nattr_mean = np.mean(np.float64(m_nattr_r), axis = 1)

    print "Male Total Mean Attr : ", m_attr_mean - tot_mean
    print "Male Total Mean Marg : ", m_marg_mean - tot_mean
    print "Male Total Mean NAttr : ", m_nattr_mean - tot_mean
    Sb = np.dot((m_attr_mean - tot_mean),np.transpose((m_attr_mean - tot_mean)))  * np.float64(m_priors[0])
    Sb += np.dot((m_marg_mean - tot_mean),np.transpose((m_marg_mean - tot_mean))) * np.float64(m_priors[1])
    Sb += np.dot((m_nattr_mean - tot_mean),np.transpose((m_nattr_mean - tot_mean))) * np.float64(m_priors[2])
    print "Male Covariance : ", Sb

    f_attr_mean = np.mean(np.float64(f_attr_r), axis = 1)
    f_marg_mean = np.mean(np.float64(f_marg_r), axis = 1)
    f_nattr_mean = np.mean(np.float64(f_nattr_r), axis = 1)

    print "Female Total Mean Attr : ", f_attr_mean - tot_mean1
    print "Female Total Mean Marg : ", f_marg_mean - tot_mean1
    print "Female Total Mean NAttr : ", f_nattr_mean - tot_mean1
    Sb1 = np.dot((f_attr_mean - tot_mean1),np.transpose((f_attr_mean - tot_mean1))) * np.float64(f_priors[0])
    Sb1 += np.dot((f_marg_mean - tot_mean1),np.transpose((f_marg_mean - tot_mean1))) * np.float64(f_priors[1])
    Sb1 += np.dot((f_nattr_mean - tot_mean1),np.transpose((f_nattr_mean - tot_mean1))) * np.float64(f_priors[2])
    print "Female Covariance : ", Sb1

    print "Sw Male Shape : ", Sw.shape
    print "Sw Female Shape : ", Sw1.shape

    print "Sb Male Shape : ", Sb.shape
    print "Sb Female Shape : ", Sb1.shape

    male_lda = LDACompute.LDA(mt_arr_r,Sw,Sb)
    male_lda.perform()
    male_fischer = male_lda.getFisherFaces((1, 10), male_pca, 100)
    print "Male LDA : ", len(male_lda.eigVal)

    female_lda = LDACompute.LDA(ft_arr_r,Sw1,Sb1)
    female_lda.perform()
    female_fischer = female_lda.getFisherFaces((1, 10), female_pca, 100)
    print "Female LDA : ", len(female_lda.eigVal)

    plot3a(male_fischer, female_fischer)

    LDA_MAX = 50
    male_acc = []
    fem_acc = []
    lda_vals = np.arange(1, LDA_MAX + 1)

    for dimension in range(1,LDA_MAX + 1):
        print "Number of LDA Features : ", dimension
        m_r_tra = male_lda.reduceFeatureDimension(dimension)
        f_r_tra = female_lda.reduceFeatureDimension(dimension)

        print "Reduced training data feature dimension"

        m_r_tst = male_lda.projectDataOntoLDA(dimension, mtst_arr)
        f_r_tst = female_lda.projectDataOntoLDA(dimension, ftst_arr)

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
    print "Male Accuracy Length : ", len(male_acc)
    print "Female Accuracy Length : ", len(fem_acc)
    print "lda values : ", lda_vals.shape
    plot3b(male_acc, fem_acc, lda_vals)