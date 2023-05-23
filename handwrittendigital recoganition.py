import numpy as np
from sklearn .datasets import load_digits
from sklearn.model_selection import train_test_split



dat=load_digits()

print(dat.target)
print(dat.data.shape)
imglength=len(dat.images)
print(imglength)

#visualize the data

import matplotlib.pyplot as plt

n=int(input("enter no under 1797 and see digite:-"))

plt.gray()
plt.matshow(dat.images[n])
plt.show()

#find x and y

x=dat.images.reshape((imglength,-1))
print(x)
y=dat.target
print


#train_test_splite



x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
print(x_train.shape)
print(x_test.shape)


# modling

from sklearn import svm

model=svm.svc(gamma=0.001)
model.fit(x_train,y_train)

#predict

pred=model.predict(x_test)



#output
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,pred)*100)





