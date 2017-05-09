from sklearn import metrics
from sklearn.metrics import roc_auc_score
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

print "Model fitting"
from sklearn.svm import SVC
#Using Linear Model for SVM
clf=SVC(kernel="linear")
#Model fitting based on training data
clf.fit(features_train,labels_train)
#Prediction based on the model
pred=clf.predict(features_test)

#Experiment 1
print "Experiment 1"
from sklearn.metrics import accuracy_score
acc=accuracy_score(pred,labels_test)

scores_test=clf.decision_function(features_test)
fpr,tpr,thresholds=metrics.roc_curve(labels_test, scores_test)
auc=roc_auc_score(labels_test, scores_test)

pyplot.figure(1)
pyplot.plot(fpr, tpr, color='blue', label='ROC Curve\n(area under curve = %f)' %auc,  lw=2)
pyplot.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
pyplot.xlabel('\nFalse Positive Rate\n', size=18)
pyplot.ylabel('\nTrue Positive Rate\n')
pyplot.title('Spambase Database Classified With Linear SVM\n')
pyplot.legend(loc='lower right')

#Experiment 2
print "Experiment 2"
coefs=numpy.copy(clf.coef_)
i=numpy.argmax(coefs)
features_train_max=numpy.array(features_train[:,i],ndmin=2).T
features_test_max=numpy.array(features_test[:,i],ndmin=2).T
coefs[0][i]=float('-Infinity')
m_array=[]
acc_array=[]
for m in range(2,58):
    i=numpy.argmax(coefs)
    coefs[0][i]=float('-Infinity')
    features_train_max=numpy.insert(features_train_max,0,features_train[:,i],axis=1)
    features_test_max=numpy.insert(features_test_max,0,features_test[:,i],axis=1)
    clf.fit(features_train_max,labels_train)
    pred=clf.predict(features_test_max)
    acc=accuracy_score(pred,labels_test)
    m_array.append(m)
    acc_array.append(acc)
    print "m: ",m,"\tAccuracy: ",acc

pyplot.figure(2)
pyplot.title("Feature Selection with Linear SVM")
pyplot.plot(m_array,acc_array)
pyplot.xlabel("m")
pyplot.ylabel("Accuracy")


m_array=[]
acc_array=[]
#Experiment 3
print "Experiment 3"
for m in range(2,58):
    feature_indices=numpy.random.choice(numpy.arange(57),m,replace=0)
    features_train_max=features_train[:,feature_indices]
    features_test_max=features_test[:,feature_indices]
    clf.fit(features_train_max,labels_train)
    pred=clf.predict(features_test_max)
    acc=accuracy_score(pred,labels_test)
    m_array.append(m)
    acc_array.append(acc)
    print "m: ",m,"\tAccuracy: ",acc

pyplot.figure(3)
pyplot.title("Random Feature Selection")
pyplot.plot(m_array,acc_array)
pyplot.xlabel("m")
pyplot.ylabel("Accuracy")
pyplot.show()
