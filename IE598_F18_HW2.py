from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt




iris = datasets.load_iris()
X = iris.data[:, [2,3]]
y = iris.target
print('Class labels:', np.unique(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1, stratify = y)
print('Labels counts in y:', np.bincount(y))
print('Labels counts in y_train:', np.bincount(y_train))
print('Labels counts in y_test:', np.bincount(y_test))

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

ppn = Perceptron(n_iter = 40, eta0 = 0.1, random_state = 1)
ppn.fit(X_train_std, y_train)

y_pred = ppn.predict(X_test_std)
print('Misclassified samples: %d' % (y_test != y_pred).sum())

print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
print('Accuracy: %.2f' % ppn.score(X_test_std, y_test))


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, p=2,
                           metric='minkowski')
knn.fit(X_train_std, y_train)


print('K-5 score:' ,knn.score(X_train_std, y_train))


knn = KNeighborsClassifier(n_neighbors=7, p=2,
                           metric='minkowski')
knn.fit(X_train_std, y_train)


print('K-7 score:' ,knn.score(X_train_std, y_train))


knn = KNeighborsClassifier(n_neighbors=10, p=2,
                           metric='minkowski')
knn.fit(X_train_std, y_train)


print('K-10 score:' ,knn.score(X_train_std, y_train))


knn = KNeighborsClassifier(n_neighbors=15, p=2,
                           metric='minkowski')
knn.fit(X_train_std, y_train)


print('K-15 score:' ,knn.score(X_train_std, y_train))


knn = KNeighborsClassifier(n_neighbors=20, p=2,
                           metric='minkowski')
knn.fit(X_train_std, y_train)


print('K-20 score:' ,knn.score(X_train_std, y_train))


knn = KNeighborsClassifier(n_neighbors=25, p=2,
                           metric='minkowski')
knn.fit(X_train_std, y_train)


print('K-25 score:' ,knn.score(X_train_std, y_train))


knn = KNeighborsClassifier(n_neighbors=1, p=2,
                           metric='minkowski')
knn.fit(X_train_std, y_train)


print('K-1 score:' ,knn.score(X_train_std, y_train))


print('My name is Stephanie Day.')
print('My netID is sjday2')
print('I hereby certify that I have read the University Policy on academic integrity and that I am not in violation.')


# try K=1 through K=25 and record testing accuracy
#k_range = range(1,26)
#scores = []
#for k in k_range:
 #   knn = KNeighborsClassifier(n_neighbors=k)
  #  knn.fit(X_train, y_train)
   # y_pred = knn.predict(X_test)
    #scores.append(metrics.accuracy_score(y_test, y_pred))
    