import glob
import Image
import numpy as np
import os
import sys

training_data = []
test_data = []

male_train = []
male_test = []

female_train = []
female_test = []

male_train_dict = []
female_train_dict = []

class Face:
    def __init__(self,image,farray):
        self.face_image = image
        self.face_array = farray
    face_image = None
    face_array = None
    seqid = None
    male = None
    age = None
    race = None
    attractiveness = None

def read_faces(subdir):
    count = 0
    # browsing the directory
    training = True
    male_not_attr = []
    female_not_attr = []

    male_marginal = []
    female_marginal = []

    male_attr = []
    female_attr = []

    for infile in glob.glob(os.path.join(subdir, '*.*')):
        count += 1

        if count > 755:
            training = False
        else:
            training = True

        (name,ext) = os.path.splitext(infile)
        (head, tail) = os.path.split(name)
        splits = tail.split("_")
        if len(splits) != 5:
            splits = tail.split("-")
        print "Name : ", tail
        print "Sub : ",splits

        im = Image.open(infile)
        im_arr = np.asarray(im)
        im_arr = im_arr.astype(np.float32)

        # turn an array into vector
        im_vec = np.reshape(im_arr, -1)
        im_vec = im_vec.T
        #print "Shape : ", im_vec.shape

        face = Face(im,im_vec)
        face.seqid = int(splits[0])
        face.attractiveness = int(splits[1])
        if splits[2].lower() == 'm':
            face.male = True
        else:
            face.male = False
        face.age = float(splits[3])
        face.race = int(splits[4])

        if training:
            training_data.append(face)
            if face.male:
                male_train.append(face)
                if face.attractiveness == 0:
                    male_not_attr.append(face)
                elif face.attractiveness == 1:
                    male_marginal.append(face)
                elif face.attractiveness == 2:
                    male_attr.append(face)
            else:
                female_train.append(face)
                if face.attractiveness == 0:
                    female_not_attr.append(face)
                elif face.attractiveness == 1:
                    female_marginal.append(face)
                elif face.attractiveness == 2:
                    female_attr.append(face)
        else:
            test_data.append(face)
            if face.male:
                male_test.append(face)
            else:
                female_test.append(face)
    #end For

    male_train_dict.append(male_not_attr)
    male_train_dict.append(male_marginal)
    male_train_dict.append(male_attr)

    female_train_dict.append(female_not_attr)
    female_train_dict.append(female_marginal)
    female_train_dict.append(female_attr)

    print "Total Count : ",count
    print "Training : ", len(training_data)
    print "Test : ", len(test_data)
    print "Training Male : ", len(male_train)
    print "Test Male : ", len(male_test)
    print "Training Female : ", len(female_train)
    print "Test Female : ", len(female_test)
    return

def getImageArrayFromPath(path):
    A = []  # A will store list of image vectors
    im = Image.open(path)
    im_arr = np.asarray(im)
    im_arr = im_arr.astype(np.float32)
    # turn an array into vector
    im_vec = np.reshape(im_arr, -1)
    return im_vec