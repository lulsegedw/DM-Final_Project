
# By: Lul Woreta
# Data Mining Course Project
# Computer Science Department
# Hood College, Fredrick
# 11/24/2020

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix


pd.options.display.max_columns=1000
pd.options.display.max_rows=1000
from IPython.display import display

#importing the datasets
training_dataset = pd.read_csv('KDDTrain+.txt',header=None)
training_dataset = training_dataset.iloc[:, :-1]
test_dataset = pd.read_csv('KDDTest+.txt',header=None)
test_dataset = test_dataset.iloc[:, :-1]

#Displaying a sample of the training dataset
display(training_dataset.sample(n=20))

#Print the different types of attacks
set(test_dataset[41])

# Group the attacks types into five categories: normal, DoS, probing, R2L, U2R

training_dataset[41] = training_dataset[41].replace(['apache2', 'back', 'land', 'neptune', 'mailbomb', 'pod', 'processtable', 'smurf', 'teardrop', 'udpstorm', 'worm'], 'DoS')
test_dataset[41] = test_dataset[41].replace(['apache2', 'back', 'land', 'neptune', 'mailbomb', 'pod', 'processtable', 'smurf', 'teardrop', 'udpstorm', 'worm'], 'DoS')
training_dataset[41] = training_dataset[41].replace(['ipsweep', 'mscan', 'nmap', 'portsweep', 'saint', 'satan'], 'R2L')
test_dataset[41] = test_dataset[41].replace(['ipsweep', 'mscan', 'nmap', 'portsweep', 'saint', 'satan'], 'R2L')
training_dataset[41] = training_dataset[41].replace(['buffer_overflow', 'loadmodule', 'perl', 'ps', 'rootkit', 'sqlattack', 'xterm'], 'U2R')
test_dataset[41] = test_dataset[41].replace(['buffer_overflow', 'loadmodule', 'perl', 'ps', 'rootkit', 'sqlattack', 'xterm'], 'U2R')
training_dataset[41] = training_dataset[41].replace(['ftp_write', 'guess_passwd', 'httptunnel', 'imap', 'multihop', 'named', 'phf', 'sendmail',  'snmpgetattack', 'snmpguess', 'spy', 'warezclient', 'warezmaster', 'xlock', 'xsnoop'], 'probing')
test_dataset[41] = test_dataset[41].replace(['ftp_write', 'guess_passwd', 'httptunnel', 'imap', 'multihop', 'named', 'phf', 'sendmail',  'snmpgetattack', 'snmpguess', 'spy', 'warezclient', 'warezmaster', 'xlock', 'xsnoop'], 'probing')
display(test_dataset.sample(10))

#encoding categorical data
labelencoder = LabelEncoder()
training_dataset[1] = labelencoder.fit_transform(training_dataset[1])
training_dataset[2] = labelencoder.fit_transform(training_dataset[2])
training_dataset[3] = labelencoder.fit_transform(training_dataset[3])
test_dataset[1] = labelencoder.fit_transform(test_dataset[1])
test_dataset[2] = labelencoder.fit_transform(test_dataset[2])
test_dataset[3] = labelencoder.fit_transform(test_dataset[3])

#encode the five classes as stated here
training_dataset[41].replace({"normal": 1, "DoS": 2, "probing": 3, "R2L": 4, "U2R": 5}, inplace=True)
test_dataset[41].replace({"normal": 1, "DoS": 2, "probing": 3, "R2L": 4, "U2R": 5}, inplace=True)

#A summary statistics
training_dataset.describe()

#Columns 6 almost all values are zeros
training_dataset[6].value_counts()

#Columns 7 almost all values are zeros
training_dataset[7].value_counts()

#Columns 8 almost all values are zeros
training_dataset[8].value_counts()

#Columns 10 almost all values are zeros
training_dataset[10].value_counts()

#plotting the correlation matrix
#Attributes 8, 14, 16, 18, 19, 20, 31, and 39 are shown in previous work that have minimum or none role in the detection of attacks
#This can be also checked here
fig, ax = plt.subplots(figsize=(30,15))
sn.heatmap(training_dataset.corr(), annot=True, ax=ax)
plt.show()

xtrain = training_dataset.iloc[:, :-1]
ytrain = training_dataset.iloc[:, 41]
xtest = test_dataset.iloc[:, :-1]
ytest = test_dataset.iloc[:, :41]

#feature scaling of the dataset.
sc_x = StandardScaler()
xtrain = sc_x.fit_transform(xtrain)
xtest = sc_x.fit_transform(xtest)
pd.DataFrame(xtest).head(10)

#MLP Classifier
#training with all the features
classifier = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(20,8))
classifier.fit(xtrain, ytrain)
ypred = classifier.predict(xtest)
cm = confusion_matrix(ytest, ypred)

#Printing the confusion matrix
pd.DataFrame(data=cm, columns=['1', '2', '3', '4', '5'], index=['1', '2', '3', '4', '5'])

data = [[9851, 65, 118, 24, 7], [1515, 6019, 143, 0, 2], [403, 164, 1854, 0, 0], [1789, 2, 6, 952, 5], [144, 4, 21, 8, 23]]
df = pd.DataFrame(data, columns=['1', '2', '3', '4', '5'], index=['1', '2', '3', '4', '5'])
df

cm = np.array(data)
FP = cm.sum(axis=0) - np.diag(cm)  
FN = cm.sum(axis=1) - np.diag(cm)
TP = np.diag(cm)
TN = cm.sum() - (FP + FN + TP)

FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)

# Detection rate
DR = TP/(TP+FN)
# False positive rate
FPR = FP/(FP+TN)
# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)
print("Overall Accuracy: "+str(ACC.mean()))

#Detection rate per class
pd.DataFrame(data=[DR], columns=['1', '2', '3', '4', '5'])

# False positive rate
pd.DataFrame(data=[FPR], columns=['1', '2', '3', '4', '5'])

#training after feature reduction (6, 7, 8, 10, 13, 14, 16, 18, 19, 20, 31, 39)
np.delete(xtrain, [6, 7, 8, 10, 13, 14, 16, 18, 19, 20, 31, 39], 1)
np.delete(xtest, [6, 7, 8, 10, 13, 14, 16, 18, 19, 20, 31, 39], 1)
classifier = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(20,8))
classifier.fit(xtrain, ytrain)
ypred = classifier.predict(xtest)
cm = confusion_matrix(ytest, ypred)
FP = cm.sum(axis=0) - np.diag(cm)  
FN = cm.sum(axis=1) - np.diag(cm)
TP = np.diag(cm)
TN = cm.sum() - (FP + FN + TP)
FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)
ACC = (TP+TN)/(TP+FP+FN+TN)
print("Overall Accuracy: "+str(ACC.mean()))

