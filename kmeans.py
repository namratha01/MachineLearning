"""
@author         : Namratha Basavanahalli Manjunatha
@Descriptions   : K-Means Clustering
"""

#Imports
from __future__ import division
from PIL import Image
import itertools
import numpy

no_of_clusters=10
no_of_classes=10
no_of_trials=5
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

def most_popular_class(cluster,labels):
    class_representation_in_cluster = [0 for i in range(no_of_classes)]
    total_instances = len(cluster)
    if total_instances == 0:
        return None
    for point in cluster:
        class_representation_in_cluster[int(labels[int(point)])]+=1
    most_popular_count = max(class_representation_in_cluster)
    first_most_popular_index = class_representation_in_cluster.index(most_popular_count)
    if class_representation_in_cluster.count(most_popular_count) is 1:
        return first_most_popular_index
    else:
        indices_of_tied_classes = []
        for c in class_representation_in_cluster:
            if c == most_popular_count:
                indices_of_tied_classes.append(
                    class_representation_in_cluster.index(c))
        return numpy.random.choice(indices_of_tied_classes)

def classify(centers,cluster_labels,datapoint):
    closest = closest_center(datapoint,centers)
    return cluster_labels[closest]

def create_confusion_matrix(classifications,test_data_labels):
    confusion_matrix=[[0 for i in range(no_of_classes)] for i in range(no_of_classes)]
    for label,classification in zip(test_data_labels,classifications):
        confusion_matrix[int(label)][int(classification)]+=1
    return confusion_matrix

def save_confusion_matrix(confusion_matrix):
    global no_of_clusters
    filename='confusion_matrix_%d_clusters.csv'%(no_of_clusters)
    output = open(filename,'w')
    for row in confusion_matrix:
        for col in row:
            output.write(str(col) + ',')
        output.write('\n')
    output.close()

def accuracy(confusion_matrix):
    m = numpy.array(confusion_matrix)
    return float(numpy.sum(numpy.diagonal(m))) / numpy.sum(m)

def pixel_value(value):
    value = int(numpy.floor(value))
    return value * 16


def draw_center_as_bitmap(name_prefix,center_number,center):
    img = Image.new('L', (8, 8), "black")
    center_2d = numpy.array(center).reshape(8, 8)
    for i in range(img.size[0]):
        for j in range(img.size[0]):
            img.putpixel((j, i), pixel_value(int(center_2d[i][j])))
    name = name_prefix + str(center_number) + '.png'
    img.save(name)

#
def k_means_clustering(k,no_of_trials):
    global train_data_list
    sse_array=[]
    sss_array=[]
    average_entropy_array=[]
    centers_array=[]
    clusters_array=[]

    for trial in range(no_of_trials):
        centers=initialize_centers(k) 
        optimized=False
        print "Trial: #%d"%(trial+1)
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
            #datasum=[[0 for i in range(64)] for i in range(no_of_clusters)]
            #centroids=[[0 for i in range(64)] for i in range(no_of_clusters)]
            #for idx,cluster in enumerate(clusters):
            #    for datapoint in cluster:
            #        datasum[idx]=datasum[idx]+train_data_list[datapoint]
            #    if sum(datasum[idx])!=0:
            #        centroids[idx]=numpy.true_divide(datasum[idx],len(cluster))

            centroids = []
            for cluster in clusters:
                mean_vector = numpy.array([0.0 for i in range(64)])  # sum feature
                # values
                for i in range(len(cluster)):
                    # sum the features
                    mean_vector += numpy.array((train_data_list[cluster[i]]))
                if len(cluster) > 0:
                    mean_vector /= float(len(cluster))
                centroids.append(mean_vector)

            # 5: Reassign each center to the centroid's location.
            old_centers = numpy.asarray(centers)
            centers = numpy.asarray(centroids)
            optimized = centers_check(old_centers,centers)

        sse=sum_squared_error(clusters,centers,train_data_list)
        sss=sum_squared_separation(clusters,centers)
        average_entropy = mean_entropy(clusters, train_data_labels)
        
        sse_array.append(sse)
        sss_array.append(sss)
        average_entropy_array.append(average_entropy)
        centers_array.append(centers)
        clusters_array.append(clusters)
    
    best_trial_idx=numpy.argmin(sse_array)
    best_sse=sse_array[best_trial_idx]
    print "Best Sum Squared Error: ",sse
    best_sss=sss_array[best_trial_idx]
    print "Best Sum Squared Separation: ",sss
    best_average_entropy=average_entropy_array[best_trial_idx]
    print "Best Mean Entropy: ",average_entropy
    best_centers=centers_array[best_trial_idx]
    best_clusters=clusters_array[best_trial_idx]
        
    for i in range(no_of_clusters):
        print "center: ",best_centers[i]
        draw_center_as_bitmap('exp_%d_center_'%no_of_clusters,i,best_centers[i])
    
    cluster_labels=[most_popular_class(cluster,train_data_labels) for cluster in best_clusters]
    classifications=[classify(best_centers,cluster_labels,datapoint) for datapoint in test_data_list]

    confusion_matrix=create_confusion_matrix(classifications,test_data_labels)
    save_confusion_matrix(confusion_matrix)

    accuracy_value=accuracy(confusion_matrix)
    print "Accuracy: ",accuracy_value

#Main function
def main():
    global no_of_trials
    global no_of_clusters
    load_training_data()
    load_test_data()
    #Experiment 1
    no_of_clusters=10
    k_means_clustering(no_of_clusters,no_of_trials)
    #Experiment 2
    no_of_clusters=30
    k_means_clustering(no_of_clusters,no_of_trials)


if __name__ == "__main__":
    main()


