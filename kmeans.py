"""
@author         : Namratha Basavanahalli Manjunatha
@Descriptions   : K-Means Clustering
"""

#Imports
import numpy
import sys

no_of_clusters=10
train_data_list=[]
test_data_list=[]

#Loading Training Data
def load_training_data():
    global train_data_list
    print "Loading Training data from optdigits.train"
    train_data=numpy.loadtxt('optdigits.train', delimiter=',')
    train_data_list=train_data[:,:-1]

#Loading Test Data
def load_test_data():
    global test_data_list
    print "Loading Test data from optdigits.test"
    test_data=numpy.loadtxt('optdigits.test', delimiter=',')
    test_data_list=test_data[:,:-1]

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
    #print distance_array
    min_dist=numpy.argmin(distance_array)
    #print min_dist
    return min_dist

#
def k_means_clustering(k):
    global train_data_list
    centers=initialize_centers(k) 
    #print numpy.asarray(centers).shape
    optimzed=False

    while optimzed is False:
        #Initialize list for closest center for each datapoint
        closest_centers=[]
        #Find closest center for each datapoint
        for datapoint in train_data_list:
            #print len(datapoint)
            #print datapoint
            closest_centers.append(closest_center(numpy.asarray(datapoint),numpy.asarray(centers)))
        #Initialize cluster lists
        clusters=[[] for i in range(no_of_clusters)]
        #Append datapoints to appropriate cluster lists
        for i in range(len(closest_centers)):
            clusters[closest_centers[i]].append(i)

        #print clusters
        #Compute centroid for each cluster
        datasum=[[0 for i in range(64)] for i in range(no_of_clusters)]
        for idx,cluster in enumerate(clusters):
            for datapoint in cluster:
                datasum[idx]=datasum[idx]+train_data_list[datapoint]
        #print numpy.asarray(datasum).shape   
        #print datasum
        centroids=numpy.true_divide(datasum,64)
        #print numpy.asarray(centroids).shape
        #print centroids
        sys.exit(1)


#Main function
def main():
    load_training_data()
    load_test_data()
    k_means_clustering(10)


if __name__ == "__main__":
    main()


