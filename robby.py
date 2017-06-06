# -*- coding: utf-8 -*-
"""
@author: Namratha Basavanahalli Manjunatha
@Des   : Q-Learning (Robby the Robot)
"""
#Imports
from PIL import ImageTk as itk
from PIL import Image
import numpy
import Tkinter as tk
import time
import random

def initialize_grid():
    grid=numpy.random.choice([1,2],[10,10])
    grid=numpy.concatenate((numpy.zeros((10,1)),grid,numpy.zeros((10,1))),axis=1)
    grid=numpy.concatenate((numpy.zeros((1,12)),grid,numpy.zeros((1,12))),axis=0)
    robby_loc_init=tuple(list((numpy.random.choice(numpy.arange(1,11),2))))
    return grid,robby_loc_init

def main():
    #Experiment 1
    print "Experiment 1"
    qmatrix=numpy.zeros((3**5,5))
    grid,robby_loc=initialize_grid()
    print grid
    print robby_loc

    #Experiment 2
    print "Experiment 2"


    #Experiment 3
    print "Experiment 3"


    #Experiment 4
    print "Experiment 4"


    #Experiment 5
    print "Experiment 5"




if __name__ == "__main__":
    main()
