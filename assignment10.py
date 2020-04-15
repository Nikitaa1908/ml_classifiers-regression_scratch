# ****************************************problem 1************************************************

import numpy as np
import random
def cal_prob(n):
    randwalk = [0]
    current_step=0
    current_no_of_step=[]

    li=[]
    for i in range(0,n):
        step = randwalk[-1]
        dice=random.randint(1,7)

        if((dice == 1 or dice == 2) and (current_step != 0)):
            current_step=current_step-1

        elif (dice == 3 or dice==4 or dice ==5):
            current_step=current_step+1

        elif (dice==6):
            current_step=current_step+6

        else:
            current_step=current_step+0
        # print(current_step)
        li.append(current_step)

        if dice <= 2:
            step = max(0, step - 1)

        elif dice <= 5:
            step += 1

        else:
            step = step + random.randint(1, 7)

        current_no_of_step.append(step)

    # print(current_no_of_step)
    return current_no_of_step,li
print("*********************on changing the function*************************")


def new_prob_calc(n,p=[],*args):
    li_new=[]
    pp=[]
    for x in p:
        pp.append(x)
    current_step=0
    for i in range(0,n):
        val = np.random.choice(np.arange(1, 7),p= pp)
        if ((val == 1 or val == 2) and (current_step != 0)):
            current_step = current_step - 1

        elif (val == 3 or val == 4 or val == 5):
            current_step = current_step + 1

        elif (val == 6):
            current_step = current_step + 6

        else:
            current_step = current_step + 0
            # print(current_step)
        li_new.append(current_step)
    # print(li_new)

prob=[0.1, 0.05, 0.05, 0.2, 0.4, 0.2]
new_prob_calc(250,prob)


# *********************************problem 2***********************************************


import pandas as pd
import scipy

# *********************************generating random dataset for logistic regression****************************************************
def generate_logistic():
    n_features = 4
    X = []
    for i in range(n_features):
      X_i = scipy.stats.norm.rvs(0, 1, 100)
      X.append(X_i)
    #print(X)
    a1 = (np.exp(1 + (0.5 * X[0]) + (0.4 * X[1]) + (0.3 * X[2]) + (0.5 * X[3]))/(1 + np.exp(1 + (0.5 * X[0]) + (0.4 * X[1]) + (0.3 * X[2]) + (0.5 * X[3]))))
    #print(a1)
    y1 = []
    for i in a1:
      if (i>=0.5):
        y1.append(1)
      else:
        y1.append(0)
    #print(y1)
    data_lr = {'X0': X[0],'X1':X[1],'X2':X[2],'X3':X[3],'Y': y1 }
    return data_lr

data=generate_logistic()
df_logistic = pd.DataFrame(data)
X_log1=df_logistic.iloc[:,0]
X_log2=df_logistic.iloc[:,1]
X_log3=df_logistic.iloc[:,2]
X_log4=df_logistic.iloc[:,3]
y_log=df_logistic.iloc[:,4]
    # print(labels)






# **********************************generating random dataset for linear regression*******************************************************
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import pandas as pd
def generate_linear(n):
    x = []
    y = []
    random_x1 = np.random.rand()
    random_x2 = np.random.rand()
    for i in range(n):
        x1 = i
        x2 = i / 2 + np.random.rand() * n
        x.append(x1+x2)
        y.append(random_x1 * x1 + random_x2 * x2 + 1)
    return np.array(x), np.array(y)


x, y = generate_linear(200)
df_linear = pd.DataFrame(list(zip(x,y)),columns =['X', 'Y'])
X_lin=df_linear.iloc[:,0]
y_lin=df_linear.iloc[:,1]
# create a linear model and fit it to the data







# ******************************************************generating the random dataset for k means clustering***************************

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
from sklearn.datasets.samples_generator import make_blobs

def generate_kmeans(n,c,std,random_state):
    X, y_true = make_blobs(n, c,std, random_state)

    plt.scatter(X[:, 0], X[:, 1], s=50)
    return X, y_true
# plt.show()


# ********************************************************problem 3 a) linear regression gradient descent***************************
import pandas as pd

plt.plot(X_lin,y_lin)
# plt.show()



m = 0
c = 0

L = 0.0001  # The learning Rate
epochs = 1000  # The number of iterations to perform gradient descent

n = float(len(X_lin))

# Gradient Descent
for i in range(epochs):
    y_pred = m * X_lin + c
    D_m = (-2 / n) * sum(X_lin * (X_lin - y_pred))
    D_c = (-2 / n) * sum(y_lin - y_pred)
    m = m - L * D_m
    c = c - L * D_c

print(m, c)

y_pred = m*X_lin + c

plt.plot(X_lin, y_lin)
plt.plot([min(X_lin), max(X_lin)], [min(y_pred), max(y_pred)], color='red')  # regression line
plt.show()

# ******************************************************************3 b) logistic regression with gradient descent****************************
X_log_final=X_log1+X_log2+X_log4-X_log3




def sigmoid(Z):
  return 1 /(1+np.exp(-Z))

def loss(y2,y_hat):
  return -np.mean(y2*np.log(y_hat) + (1-y2)*(np.log(1-y_hat)))

W = np.zeros((4,1))
b = np.zeros((1,1))

m = len(y_log)
lr = 0.001
for epoch in range(1000):
  Z = np.matmul(X_log_final,W)+b
  A = sigmoid(Z)
  logistic_loss = loss(y_log,A)
  dz = A - y_log
  dw = 1/m * np.matmul(X_log_final.T,dz)
  db = np.sum(dz)

  W = W - lr*dw
  b = b - lr*db

  if epoch % 100 == 0:
    print(logistic_loss)



# ****************************************3 c) linear regression with l1 and l2 regularization**************
# l1 regularization

b0 = 0
b1 = 0
l = 0.001
epochs = 200
lamda = 0.1

n = float(len(X_lin))
for i in range(epochs):
    y_p = b1 * X_lin + b0
    loss = np.sum(y_p - X_lin) ** 2 + (lamda * b1)
    d1 = (-2 / n) * sum(X_lin * (y_lin - y_p)) + lamda
    d0 = (-2 / n) * sum(y_lin - y_p)
    b1 = b1 - (l * d1)
    b0 = b0 - (l * d0)

print(b1, b0)



# l2 regularization

c0 = 0
c1 = 0
l2 = 0.001
epochs2 = 200
lamda2 = 0.1

n2 = float(len(X_lin))
for i in range(epochs2):
    y_p2 = c1 * X_lin + c0
    loss2 = np.sum(y_p2 - y_lin) ** 2 + ((lamda2 / 2) * c1)
    g1 = (-2 / n2) * sum(X_lin * (y_lin - y_p2)) + (lamda2 * c1)
    g0 = (-2 / n2) * sum(y_lin - y_p2)
    c1 = c1 - (l * g1)
    c0 = c0 - (l * g0)

print(c1, c0)




# ******************************************* 3 d) logistic regression using l1 regularization******************


lamda3 = 0.1

W2 = np.zeros((4,1))
b2 = np.zeros((1,1))
m2 = len(y_log)
lr = 0.001
for epoch in range(1000):
  Z2 = np.matmul(X_log_final,W2)+b2
  A2 = sigmoid(Z2)
  logistic_loss2 = loss(y_log,A2)
  dz2 = A2 - y_log
  dw2 = 1/m2 * np.matmul(X_log_final.T,dz2) + lamda3
  db2 = np.sum(dz2)

  W2 = W2 - lr*dw2
  b2 = b2 - lr*db2

  if epoch % 100 == 0:
    print(logistic_loss2)






# **********************************************logistic regression using l2 regularization*****************
W3 = np.zeros((4,1))
b3 = np.zeros((1,1))

for epoch in range(1000):
  Z3 = np.matmul(X_log_final,W3)+b3
  A3 = sigmoid(Z3)
  logistic_loss3 = loss(y_log,A3)
  dz3 = A3 - y_log
  dw3 = 1/m2 * np.matmul(X_log_final.T,dz3) + lamda3
  db3 = np.sum(dz3)

  W3 = W3 - lr*dw3
  b3 = b3 - lr*db3

  if epoch % 100 == 0:
    print(logistic_loss3)






# ****************************************************3 e) kmeans clustering****************************************

from copy import deepcopy
f1,f2=generate_kmeans(250, 4,0.60,0)
X5 = np.array(list(zip(f1, f2)))
plt.scatter(f1, f2, c='black', s=7)
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)

k = 2
# X coordinates of random centroids
C_x = np.random.randint(0, np.max(X5)-20, size=k)
# Y coordinates of random centroids
C_y = np.random.randint(0, np.max(X5)-20, size=k)
C = np.array(list(zip(C_x, C_y)), dtype=np.float32)
print(C)
plt.scatter(f1, f2, c='#050505', s=7)
plt.scatter(C_x, C_y, marker='*', s=200, c='g')

C_old = np.zeros(C.shape)
# Cluster Lables(0, 1, 2)
clusters = np.zeros(len(X5))
# Error func. - Distance between new centroids and old centroids
error = dist(C, C_old, None)
# Loop will run till the error becomes zero
while error != 0:
    # Assigning each value to its closest cluster
    for i in range(len(X5)):
        distances = dist(X5[i], C)
        cluster = np.argmin(distances)
        clusters[i] = cluster
    # Storing the old centroid values
    C_old = deepcopy(C)
    # Finding the new centroids by taking the average value
    for i in range(k):
        points = [X5[j] for j in range(len(X5)) if clusters[j] == i]
        C[i] = np.mean(points, axis=0)
    error = dist(C, C_old, None)

colors = ['r', 'g', 'b', 'y', 'c', 'm']
fig, ax = plt.subplots()
for i in range(k):
        points = np.array([X5[j] for j in range(len(X5)) if clusters[j] == i])
        ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='#050505')





# *********************************************linear regression using oops*****************************


import numpy as np


class LinearRegressionModel():

    def __init__(self, dataset, learning_rate, num_iterations):
        self.dataset = np.array(dataset)
        self.b = 0
        self.m = 0
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.M = len(self.dataset)
        self.total_error = 0

    def apply_gradient_descent(self):
        for i in range(self.num_iterations):
            self.do_gradient_step()

    def do_gradient_step(self):
        b_summation = 0
        m_summation = 0
        for i in range(self.M):
            x_value = self.dataset[i, 0]
            y_value = self.dataset[i, 1]
            b_summation += (((self.m * x_value) + self.b) - y_value)
            m_summation += (((self.m * x_value) + self.b) - y_value) * x_value
        self.b = self.b - (self.learning_rate * (1 / self.M) * b_summation)
        self.m = self.m - (self.learning_rate * (1 / self.M) * m_summation)

    def compute_error(self):
        for i in range(self.M):
            x_value = self.dataset[i, 0]
            y_value = self.dataset[i, 1]
            self.total_error += ((self.m * x_value) + self.b) - y_value
        return self.total_error

    def __str__(self):
        return "Results: b: {}, m: {}, Final Total error: {}".format(round(self.b, 2), round(self.m, 2),
                                                                     round(self.compute_error(), 2))

    def get_prediction_based_on(self, x):
        return round(float((self.m * x) + self.b), 2)  # Type: Numpy float.


def main():
    dataset = df_linear
    lr = LinearRegressionModel(dataset, 0.0001, 1000)
    lr.apply_gradient_descent()
    hours = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for hour in hours:
        print("Studied {} hours and got {} points.".format(hour, lr.get_prediction_based_on(hour)))
    print(lr)


if __name__ == "__main__": main()



# *******************************************logistic regression using oops***********************************


class LogisticRegression:
    def __init__(self, learning_rate, num_iters, fit_intercept=True, verbose=False):
        self.learning_rate = learning_rate
        self.num_iters = num_iters
        self.fit_intercept = fit_intercept
        self.verbose = verbose

    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        self.theta = np.zeros(X.shape[1])

        for i in range(self.num_iters):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size

            self.theta -= self.learning_rate * gradient

            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            loss = self.__loss(h, y)

            if self.verbose == True and i % 1000 == 0:
                print(f'Loss: {loss}\t')

    def predict_probability(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        return self.__sigmoid(np.dot(X, self.theta))

    def predict(self, X):
        return (self.predict_probability(X).round())




# ****************************************************k means using oops**********************************

class K_Means:
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, data):

        self.centroids = {}

        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []

            for featureset in X:
                distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid - original_centroid) / original_centroid * 100.0) > self.tol:
                    print(np.sum((current_centroid - original_centroid) / original_centroid * 100.0))
                    optimized = False

            if optimized:
                break

    def predict(self, data):
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification


colors = 10 * ["g", "r", "c", "b", "k"]