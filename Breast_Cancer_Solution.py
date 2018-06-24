from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier

# Read DataSet
dt = pd.read_csv('breast_cancer.csv')
array = dt.values
# Test and Split DataSet
X = array[:,2:32]
Y = array[:,1]
seed = 6
validation_size = 0.10
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
scoring = 'accuracy'

# Random Forest Algorithm
dt = RandomForestClassifier()
dt.fit(X_train, Y_train)
predictions = dt.predict(X_test)

# Confusion Matrix
confusion=confusion_matrix(Y_test,predictions)
print(confusion)
print(accuracy_score(Y_test,predictions))
print(classification_report(Y_test,predictions))

model =ExtraTreesClassifier()
model.fit(X_train,Y_train)

# True Positives
TP = confusion[1, 1]
# True Negatives
TN = confusion[0, 0]
# False Positives
FP = confusion[0, 1]
# False Negatives
FN = confusion[1, 0]

print("FP=",FP)
print("FN=",FN)
print("TP=",TP)
print("TN=",TN)
print("ACCURACY={}".format((TP + TN) / float(TP + TN + FP + FN)))
print(accuracy_score(Y_test,predictions))
print("FPR=",FP / float(TN + FP))
print("TPR=",TP / float(TP + FN))
print("TNR=",TN/float(TN+FP))
print("FNR=",FN/float(FN+TP))
