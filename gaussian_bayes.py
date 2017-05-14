from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
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



