# -*- coding: utf-8 -*-
"""
@author: Namratha Basavanahalli Manjunatha
@Des   : Perceptrons
"""
#Imports
from __future__ import division
import numpy
import scipy.special
import matplotlib.pyplot as plt


class neuralNetwork:

    #Constructor
    def __init__(self,inputNodes,outputNodes,learningRate,epoch):
        #Setting parameters
        self.inodes=inputNodes
        self.onodes=outputNodes
        #Learning Rate
        self.lrate=learningRate
        self.nepochs=epoch
        #Weight between Input and Output
        self.wio=numpy.random.choice([-0.05,0.05],(self.onodes,self.inodes))
        #Apply activation function
        self.activation_function= lambda x:scipy.special.expit(x)
        pass
    
    #Train Network
    def train(self,inputs_list,targets_list):

        #Calculate Network Outputs

        #Convert into a 2D Array
        inputs=numpy.array(inputs_list,ndmin=2).T
        targets=numpy.array(targets_list,ndmin=2).T
        #Calculate signal into output layer
        final_inputs=numpy.dot(self.wio,inputs)
        #Pass final inputs through activation function
        final_outputs=self.activation_function(final_inputs)

        #Calculate Error
        output_errors=targets-final_outputs

        #Update Weights between output and input
        self.wio+=self.lrate*numpy.dot((output_errors*final_outputs*(1-final_outputs)),numpy.transpose(inputs))
        pass
    
    #Score with Network
    def query(self,inputs_list):
        #Convert input into a 2D Array
        inputs=numpy.array(inputs_list,ndmin=2).T
        #Calculate dot product for final layer
        final_inputs=numpy.dot(self.wio,inputs)
        #Pass final inputs through activation function
        final_outputs=self.activation_function(final_inputs)
        #print (final_outputs)
        return final_outputs
        pass


#Network Parameters
input_nodes=785
output_nodes=10
learning_rates=[0.1,0.01,0.001]
epoch=5
#test_perf_array=[]
#test_epoch_array=[]
#train_perf_array=[]
#train_epoch_array=[]
i=220

for learning_rate in learning_rates:
    test_perf_array=[]
    test_epoch_array=[]
    train_perf_array=[]
    train_epoch_array=[]

    #Create a sample network
    ANetwork=neuralNetwork(input_nodes,output_nodes,learning_rate,epoch)
    
    for ep in range(epoch):
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
            inputs=(numpy.asfarray(all_values[1:])/255.0)
            inputs = numpy.append(inputs,[1])
            #Setup target values
            targets=numpy.zeros(output_nodes)
            targets[int(all_values[0])]=0.99
            ANetwork.train(inputs,targets)

        #Compute accuracy for test data
        test_data_file=open("mnist_test.csv","r")
        test_data_list=test_data_file.readlines()
        test_data_file.close()

        test_scorecard=[]
        
        print(ep,"epoch")
        for record in test_data_list:
            all_values=record.split(',')
            correct_label=int(all_values[0])
            #print(correct_label,"correct label")
            inputs=(numpy.asfarray(all_values[1:])/255.0)
            inputs = numpy.append(inputs,[1])
            outputs=ANetwork.query(inputs)
            label=numpy.argmax(outputs)
            #print(label,"Network Answer")
            if(label==correct_label):
                test_scorecard.append(1)
            else:
                test_scorecard.append(0)

        test_scorecard_array=numpy.asarray(test_scorecard)
        print (test_scorecard_array)
        test_sumval=test_scorecard_array.sum()
        test_size=test_scorecard_array.size
        test_perf=float(test_sumval/test_size)*100;
        
        test_perf_array.append(test_perf)
        test_epoch_array.append(ep)

        print("Test Data Performance="+str(test_perf)+"%")
   

        #Compute accuracy for training data
        test_data_file=open("mnist_train.csv","r")
        test_data_list=test_data_file.readlines()
        test_data_file.close()

        train_scorecard=[]
        
        print(ep,"epoch")
        for record in test_data_list:
            all_values=record.split(',')
            correct_label=int(all_values[0])
            #print(correct_label,"correct label")
            inputs=(numpy.asfarray(all_values[1:])/255.0)
            inputs = numpy.append(inputs,[1])
            outputs=ANetwork.query(inputs)
            label=numpy.argmax(outputs)
            #print(label,"Network Answer")
            if(label==correct_label):
                train_scorecard.append(1)
            else:
                train_scorecard.append(0)

        train_scorecard_array=numpy.asarray(train_scorecard)
        print (train_scorecard_array)
        train_sumval=train_scorecard_array.sum()
        train_size=train_scorecard_array.size
        train_perf=float(train_sumval/train_size)*100;
        
        train_perf_array.append(train_perf)
        train_epoch_array.append(ep)

        print("Training Data Performance="+str(train_perf)+"%")


    i+=1
    plt.figure(1)
    plt.subplot(i)
    plt.title("Learning Rate: %s"%learning_rate)
    plt.plot(test_epoch_array,test_perf_array,'b')
    plt.plot(train_epoch_array,train_perf_array,'r')
    plt.legend()
    plt.ylabel("Accuracy %")
    plt.xlabel("Epoch")
    plt.tight_layout()

plt.show()
print "ALL DONE!"



