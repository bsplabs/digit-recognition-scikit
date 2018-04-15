import cv2
import numpy as np
from sklearn import neighbors

img = cv2.imread("digit.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = np.array(img)

data_set = [np.vsplit(row,50) for row in np.hsplit(img,100)]
data_set = np.array(data_set).reshape(-1,400)

X_train = data_set[:2500]
X_test = data_set[2500:]

Y_train = np.tile(np.repeat(np.arange(10),5),50)

classifier = neighbors.KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train, Y_train)

predicted = classifier.predict(X_test[20])
print(predicted)

cv2.imshow(str(predicted), X_test[20].reshape(20,20))
cv2.waitKey()