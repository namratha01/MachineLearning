"""
@author         : Namratha Basavanahalli Manjunatha
@Descriptions   : K-Means Clustering
"""

#Imports
import numpy


#Loading Training Data
def load_training_data(self):
    print "Loading Training data from optdigits.train"
    train_data=open("optdigits.train","r")
    train_data_list=train_data.readlines()
    train_data.close()

#Loading Test Data
def load_test_data(self):
    print "Loading Test data from optdigits.test"
    test_data=open("optdigits.test","r")
    test_data_list=test_data.readlines()
    test_data.close()

#Initialize center 
def initialize_centers(k):
    return [numpy.random.randint(0,16,64).tolist() for i in range(k)]

#Compute distance between datapoint and center
def distance(datapoint,center):
    sub=datapoint-center
    sq=numpy.square(sub)
    return numpy.sqrt(sum(sq))

#Find the closest center to the datapoint
def closest_center(datapoint,centers,k=10):
    distance_array=[distance(datapoint,centers[i]) for i in range(0,k)]
    print distance_array
    min_dist=numpy.argmin(distance_array)
    print min_dist


#
def k_means_clustering(k):
   centers=initialize_centers(k) 

#Main function
def main():
    load_training_data()
    load_test_data()
    k_means_clustering(10)


if __name__ == "__main__":
        main()


