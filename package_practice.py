# import csv
# with open('train.csv') as f:
#     myCsv = csv.reader(f)
#     headers = next(myCsv)
#     for row in myCsv:
#         print(row)
    

from sklearn import svm
X = [[0, 0, 0], [1, 1, 1]]
y = [0, 1]
clf = svm.SVC()
clf.fit(X, y)
#SVC()
prediction = clf.predict([[2, 2, 2]])

print(prediction)