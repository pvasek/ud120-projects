#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
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


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#features_train = features_train[:len(features_train)/100] 
#labels_train = labels_train[:len(labels_train)/100] 

for c in [10000]:
    clf = SVC(kernel='rbf', C=c)

    training_time = time()
    clf.fit(features_train, labels_train)
    training_time = time()-training_time

    predict_time = time()
    prediction_test = clf.predict(features_test)
    predict_time = time()-predict_time

    accuracy_time = time()
    accuracy = accuracy_score(labels_test, prediction_test)
    accuracy_time = time()-accuracy_time

    print("c: {}, accuracy: {}, training_time: {}, predict_time: {}, accuracy_time: {}".format(c, accuracy, training_time, predict_time, accuracy_time))

    print "Cris emails: ", len(filter(lambda x: x == 1, prediction_test))
