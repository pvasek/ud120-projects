#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
#plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary


from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

# best = 0
# for n_estimators in range(1, 100, 5):
#     for learning_rate in [1, 2, 5, 10, 100]:
#         clf = AdaBoostClassifier(n_estimators = n_estimators, learning_rate = learning_rate)
#         clf = clf.fit(features_train, labels_train)
#         prediction_test = clf.predict(features_test)
#         accuracy = accuracy_score(labels_test, prediction_test)
#         if accuracy > best:
#             best = accuracy
#             print "n_estimators: ", n_estimators, "learning_rate: ", learning_rate, ", accuracy: ", accuracy

# best results:
# n_estimators:  16 learning_rate:  1 , accuracy:  0.928

clf = AdaBoostClassifier(n_estimators = 16, learning_rate = 1)
clf = clf.fit(features_train, labels_train)
prediction_test = clf.predict(features_test)
accuracy = accuracy_score(labels_test, prediction_test)
print "accuracy: ", accuracy

#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.gcf().canvas.set_window_title("Ada boost")
prettyPicture(clf, features_test, labels_test)
plt.show()