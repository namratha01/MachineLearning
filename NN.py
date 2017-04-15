# -*- coding: utf-8 -*-
"""
@author: Namratha Basavanahalli Manjunatha
@Des   : Perceptrons
"""
#Imports
from __future__ import division
import numpy
import scipy.special
import matplotlib.pyplot


class neuralNetwork:

    #Constructor
    def __init__(self,inputNodes,outputNodes,learningRate):
        #Setting parameters
        self.inodes=inputNodes
        self.onodes=outputNodes
        #Learning Rate
        self.lrate=learningRate
        #Weight between Input and Output
        self.wio=numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.inodes))
        #Apply activation function
        self.activation_function= lambda x:scipy.special.expit(x)
        pass
    #Train Network
    def train(self,inputs_list,targets_list):

        #Calculate Network Outputs

        #Convert into a 2D Array
        inputs=numpy.array(inputs_list,ndmin=2).T
        targets=numpy.array(targets_list,ndmin=2).T
        #Calculate signals into hidden layer
        #Pass inputs through actiavation function
        outputs=self.activation_function(inputs)
        #Calculate signal into output layer
        final_inputs=numpy.dot(self.wio,outputs)
        #Pass final inputs through activation function
        final_outputs=self.activation_function(final_inputs)

        #Calculate Error
        output_errors=targets-final_outputs

        #Update Weights between output and hidden layer
        self.wio+=self.lrate*numpy.dot((output_errors*final_outputs*(1-final_outputs)),numpy.transpose(inputs))


        pass
    #Score with Network
    def query(self,inputs_list):
        #Convert input into a 2D Array
        inputs=numpy.array(inputs_list,ndmin=2).T
        #Calulate dot product for hidden signal input
        #Pass hidden signals through activation function
        outputs=self.activation_function(inputs)
        #Calculate dot product for final layer
        final_inputs=numpy.dot(self.wio,outputs)
        #Pass final inputs through activation function
        final_outputs=self.activation_function(final_inputs)
        print (final_outputs)
        return final_outputs



#Network Parameters
input_nodes=784
output_nodes=10
learning_rate=0.1
epoch=50


#Create a sample network
ANetwork=neuralNetwork(input_nodes,output_nodes,learning_rate)

#Load training data
training_data_file=open("mnist_train.csv","r")
training_data_list=training_data_file.readlines()
training_data_file.close()

#Train the network
#1. Go through all records in the training data set
for record in training_data_list:
    #Split by comma
    all_values=record.split(",")
    #Normalize the values
    inputs=(numpy.asfarray(all_values[1:])/255.0*0.99)+0.01
    #Setup target values
    targets=numpy.zeros(output_nodes)+0.01
    targets[int(all_values[0])]=0.99
    ANetwork.train(inputs,targets)

#Test the network
test_data_file=open("mnist_test.csv","r")
test_data_list=test_data_file.readlines()
test_data_file.close()

scorecard=[]

for record in test_data_list:
    all_values=record.split(',')
    correct_label=int(all_values[0])
    print(correct_label,"correct label")
    inputs=(numpy.asfarray(all_values[1:])/255.0*0.99)+0.01
    outputs=ANetwork.query(inputs)
    label=numpy.argmax(outputs)
    print(label,"Network Answer")
    if(label==correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)

scorecard_array=numpy.asarray(scorecard)
print (scorecard_array)
sumval=scorecard_array.sum()
size=scorecard_array.size
perf=float(sumval/size)*100;

print("Performance="+str(perf)+"%")
