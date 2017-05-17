from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
import numpy

print "Loading data from Spambase"
spam_data=shuffle(numpy.loadtxt('spambase.data.csv', delimiter=','))
features_unscaled=spam_data[:,:-1]
features=preprocessing.StandardScaler().fit_transform(features_unscaled)
labels=spam_data[:,57]

print "Splitting SPAM data into training and test set"
features_train, features_test, labels_train, labels_test = train_test_split(features,labels,test_size=0.50)
indices=[numpy.nonzero(labels_train==0)[0],numpy.nonzero(labels_train)[0]]
mean=numpy.transpose([numpy.mean(features_train[indices[0],:],axis=0),numpy.mean(features_train[indices[1],:],axis=0)])
std=numpy.transpose([numpy.std(features_train[indices[0],:],axis=0),numpy.std(features_train[indices[1],:],axis=0)])


#PART I - Classification with Naive Bayes


#PART II - Classification with Logistic Regression
lrm=LogisticRegression()
lrm=lrm.fit(features_train,labels_train)
pred=lrm.predict(features_test)
acc=accuracy_score(pred,labels_test)
#Compute Precision, Recall
print "\nClassification Report"
print classification_report(labels_test,pred)
print "Confusion matrix"
print confusion_matrix(labels_test,pred)
print "Accuracy: ",acc








