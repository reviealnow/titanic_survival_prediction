import csv
from sklearn import svm
features = []
labels = []
with open('train.csv') as f:
    myCsv = csv.reader(f)
    headers = next(myCsv)
    # PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
    sumAge = 0
    lengAge = 0
    # averageAge = sumAge/lengAge
    for row in myCsv:
        # print(row)
        PassengerId = row[0]
        Survived = row[1]
        Pclass = row[2]
        Name = row[3]
        Sex = row[4]
        Age = row[5]
        SibSp = row[6]
        Parch = row[7]
        Ticket = row[8]
        Fare = row[9]
        Cabin = row[10]
        Embarked = row[11]
        
        if Sex=="male":
            Sex=0
        elif Sex=="female":
            Sex=1

        if Embarked=="C":
            Embarked=1
        elif Embarked=="Q":
            Embarked=2
        elif Embarked=="S":
            Embarked=3
        else:
            Embarked=4
        
        if Age == '':
            Age = 0
        else:
            sumAge = sumAge + float(Age)
            lengAge = lengAge + 1

        labels.append(float(Survived))
        features.append([float(Pclass),Sex,float(Age),float(SibSp),float(Parch),float(Fare),Embarked])

print(labels)
averageAge = sumAge/lengAge
for feature in features:
    if feature[2] == 0:
        feature[2] = averageAge
    
# print(features)

from sklearn import svm
import numpy as np

X = np.array(features)
y = np.array(labels)
clf = svm.SVC(kernel="rbf", gamma="auto") 
clf.fit(X, y)
prediction = clf.predict([[3.0, 1, 5, 8.0, 2.0, 69.55, 3]])
print(prediction)
print(clf.score(X,y))

#//
label_test = []
features_test = []
PassengerIds = []

with open('test.csv') as f:
    myCsv = csv.reader(f)
    headers = next(myCsv)
# [PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked]
    sumAge = 0 
    lengAge = 0
    sumFare = 0
    lengFare = 0
    # averageAge = sumAge/lengAge
    for row in myCsv:
        # print(row)
        PassengerId = row[0]
        Pclass = row[1]
        Name = row[2]
        Sex = row[3]
        Age = row[4]
        SibSp = row[5]
        Parch = row[6]
        Ticket = row[7]
        Fare = row[8]
        Cabin = row[9]
        Embarked = row[10]
        
        PassengerIds.append(PassengerId)

        if Sex=="male":
            Sex=0
        elif Sex=="female":
            Sex=1

        if Embarked=="C":
            Embarked=1
        elif Embarked=="Q":
            Embarked=2
        elif Embarked=="S":
            Embarked=3
        else:
            Embarked=4
        
        if Age == '':
            Age = 0
        else:
            sumAge = sumAge + float(Age)
            lengAge = lengAge + 1
        
        if Fare == '':
            Fare = 0
        else:
            sumFare = sumFare + float(Fare)
            lengFare = lengFare + 1

        # labels.append(Survived)
        # features.append([float(Pclass),Sex,float(Age),float(SibSp),float(Parch),float(Fare),Embarked])
        features_test.append([float(Pclass),Sex,float(Age),float(SibSp),float(Parch),float(Fare),Embarked])

averageAge_test = sumAge/lengAge
averageFare_test = sumFare/lengFare

for feature in features_test:
    if feature[2] == 0:
        feature[2] = averageAge_test

    if feature[5] == 0:
        feature[5] = averageAge_test
    
# print(features_test)

predictionList = clf.predict(features_test)

print(predictionList)
print(PassengerIds)

with open('gender_submission.csv', 'w', newline='') as csvfile:
    #建立 CSV 檔案寫入器
    writer = csv.writer(csvfile)

    #寫入一列資料
    writer.writerow(['PassengerId', 'Survived'])
    
    # 寫入另外幾列資料
    n = 0 
    for passengerId in PassengerIds:
        writer.writerow([passengerId,predictionList[n]])
        n = n+1
