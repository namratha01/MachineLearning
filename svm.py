from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
import numpy

spam_data = shuffle(numpy.loadtxt('spambase.data.csv', delimiter=','))
features_unscaled=spam_data[:,:-1]
features=preprocessing.scale(features_unscaled)
labels=spam_data[:,57]

features_train, features_test, labels_train, labels_test = train_test_split(features,labels,test_size=0.50)

from sklearn.svm import SVC
clf=SVC(kernel="linear")
clf.fit(features_train,labels_train)
pred=clf.predict(features_test)

from sklearn.metrics import accuracy_score
acc=accuracy_score(pred,labels_test)

scores_test=clf.decision_function(features_test)
fpr, tpr, thresholds = metrics.roc_curve(labels_test, scores_test)
auc = roc_auc_score(labels_test, scores_test)

pyplot.figure(figsize=(8, 7.5), dpi=100)
pyplot.plot(fpr, tpr, color='blue', label='ROC Curve\n(area under curve = %f)' %auc,  lw=2)
pyplot.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
pyplot.xlabel('\nFalse Positive Rate\n', size=18)
pyplot.ylabel('\nTrue Positive Rate\n', size=18)
pyplot.title('Spambase Database Classified With Linear SVM\n', size=20)
pyplot.legend(loc='lower right')
pyplot.show()

print "Accuracy: ",acc
