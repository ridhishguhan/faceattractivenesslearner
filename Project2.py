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
import Task1
import Task2
import Task3
import Task4

data_path = "E:\\EE5907R\\project2\\project2_faces"
plot_save_path = "E:\\EE5907R\\project2\\"

def main():
    #data_path = raw_input("Enter path : ")
    DataSet.read_faces(data_path)
    print 2*"\n*******************************************"

#project sequence
main()
#Task1.task1()
#Task2.task2()
#Task3.task3()
Task4.task4()