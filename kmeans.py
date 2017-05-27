"""
@author         : Namratha Basavanahalli Manjunatha
@Descriptions   : K-Means Clustering
"""

#Imports
from __future__ import division
import itertools
import numpy
import sys

no_of_clusters=10
no_of_classes=10
train_data_list=[]
test_data_list=[]
train_data_labels=[]
test_data_labels=[]

#Loading Training Data
def load_training_data():
    global train_data_list
    global train_data_labels
    print "Loading Training data from optdigits.train"
    train_data=numpy.loadtxt('optdigits.train', delimiter=',')
    train_data_list=train_data[:,:-1]
    train_data_labels=train_data[:,-1]

#Loading Test Data
def load_test_data():
    global test_data_list
    global test_data_labels
    print "Loading Test data from optdigits.test"
    test_data=numpy.loadtxt('optdigits.test', delimiter=',')
    test_data_list=test_data[:,:-1]
    test_data_labels=test_data[:,-1]

#Initialize center 
def initialize_centers(k):
    return [numpy.random.randint(0,16,64).tolist() for i in range(k)]

#Compute distance between datapoint and center
def distance(datapoint,center):
    sub=datapoint-center
    sq=numpy.square(sub)
    return numpy.sqrt(sum(sq))

#Find the closest center to the datapoint
def closest_center(datapoint,centers,no_of_clusters=10):
    distance_array = list()
    for center in centers:
        distance_array.append(distance(datapoint, center))
    first_min_distance = numpy.argmin(distance_array)

    min_distances = list()
    for i in range(len(distance_array)):
        if distance_array[i] - distance_array[first_min_distance] < 10 ** -10:
            min_distances.append(i)
    return numpy.random.choice(min_distances)

def centers_check(old_centers, centers):
    difference = 0
    for old_center, new_center in zip(old_centers, centers):
        difference += numpy.sum(numpy.abs(numpy.array(old_center) - numpy.array(new_center)))
    #print difference
    if difference < 10 ** -1:
        return True
    else:
        return False

def sum_squared_error(clusters,centers,train_data_list):
    error=0
    for center,cluster in zip(centers,clusters):
        for datapoint_idx in cluster:
            datapoint=train_data_list[datapoint_idx]
            error+=distance(datapoint,center)**2
    return error

def sum_squared_separation(clusters,centers):
    pairs=itertools.combinations([i for i in range(no_of_clusters)],2)
    separation=0
    for pair in pairs:
        separation+=distance(centers[pair[0]],centers[pair[1]])**2
    return separation

def entropy(cluster,labels):
    global no_of_classes
    entropy_sum=0
    class_representation_in_cluster = [0 for i in range(no_of_classes)]
    total_instances = len(cluster)

    if total_instances == 0:
        return 0
    for datapoint in cluster:
        class_representation_in_cluster[int(labels[datapoint])] += 1
    class_ratios = [float(class_representation_in_cluster[i]) / total_instances
                    for i in range(no_of_classes)]
    for i in range(no_of_classes):
        if class_representation_in_cluster[i] < 1:
            product = 0.0
        else:
            product = class_ratios[i] * numpy.log2(class_ratios[i])
        entropy_sum += product
    return -1 * entropy_sum

def mean_entropy(clusters,labels):
    datapoint_per_cluster=[len(cluster) for cluster in clusters]
    no_of_datapoints=sum(datapoint_per_cluster)
    ratio_array=[float(datapoint_per_cluster[i])/no_of_datapoints for i in range(no_of_clusters)]
    weighted_entropies=[ratio_array[i]*entropy(clusters[i],labels) for i in range(no_of_clusters)]
    mean=float(sum(weighted_entropies))/len(weighted_entropies)
    return mean

#
def k_means_clustering(k):
    global train_data_list
    centers=initialize_centers(k) 
    optimized=False

    while optimized is False:
        #Initialize list for closest center for each datapoint
        closest_centers=[]
        #Find closest center for each datapoint
        for datapoint in train_data_list:
            closest_centers.append(closest_center(numpy.asarray(datapoint),numpy.asarray(centers)))
        #Initialize cluster lists
        clusters=[[] for i in range(k)]
        #Append datapoints to appropriate cluster lists
        for i in range(len(closest_centers)):
            clusters[closest_centers[i]].append(i)

        #Centroid Computation
        centroids = []
        for cluster in clusters:
            mean_vector = numpy.array([0.0 for i in range(64)])
            for i in range(len(cluster)):
                mean_vector += numpy.array((train_data_list[cluster[i]]))
            if len(cluster) > 0:
                mean_vector /= float(len(cluster))
            centroids.append(mean_vector)

        # 5: Reassign each center to the centroid's location.
        old_centers = centers
        centers = centroids
        optimized = centers_check(old_centers, centers)

    sse=sum_squared_error(clusters,centers,train_data_list)
    print "Sum Squared Error: ",sse
    sss=sum_squared_separation(clusters,centers)
    print "Sum Squared Separation: ",sss
    average_entropy = mean_entropy(clusters, train_data_labels)
    print "Mean Entropy: ",average_entropy

#Main function
def main():
    load_training_data()
    load_test_data()
    k_means_clustering(10)


if __name__ == "__main__":
    main()


