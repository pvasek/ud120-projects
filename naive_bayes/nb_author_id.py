#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###


#########################################################

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

clf = GaussianNB()

t0 = time()
clf.fit(features_train, labels_train)
t0 = time()-t0
print "training time:", round(t0, 3), "s"

t1 = time()
tmp = clf.predict(features_train)
t1 = time()-t1
print "predict time for training group:", round(t1, 3), "s"

t2 = time()
prediction_test = clf.predict(features_test)
t2 = time()-t2
print "predict time:", round(t2, 3), "s"

t3 = time()
accuracy = accuracy_score(labels_test, prediction_test)
print "score time:", round(time()-t3, 3), "s"

print("accuracy: {}, fake t/p: {}, real t/p: {}".format(accuracy, t0/t2, t0/t1))


