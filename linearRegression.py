import matplotlib.pyplot as plt
import numpy as np

# import data
data = np.genfromtxt('patronage2.txt', delimiter=',');

# Put data into columns
days = data[:, 0]
patronage = data[:, 1]
patronage = patronage.reshape(len(patronage), 1)
y = patronage
m = len(days)

# Add x0 to days to make design matrix
x0 = np.ones(len(days))
X = np.vstack((x0, days))
X = X.transpose()

# Set theta and gradient descent parameters
theta = np.zeros((2, 1))
iterations = 1500
alpha = 0.01

def computeCost(theta, X, y):
    # print(X.shape)
    # print(theta.shape)
    hyp = np.dot(X, theta)
    err = hyp - y
    squaredErr = np.square(err)
    sumSquaredErr = np.sum(squaredErr)
    J = (1 / (2*m)) * sumSquaredErr
    return J

def gradientDescent(X, y, theta, alpha, iterations):
    m = len(y)
    J_history = np.zeros((iterations))

    for i in range(1, iterations):
        hyp = np.dot(X, theta)
        err = hyp - y
        gradient = (alpha/m) * np.dot(X.transpose(), err)
        theta = theta - gradient
    return  theta


def predict(x, theta):
    prediction = np.dot(x, theta)
    return prediction

theta = gradientDescent(X, patronage, theta, alpha, iterations)
# print(theta)
J = computeCost(theta, X, y)
# print(J)

predict1 = predict(np.array([1, 1.02]), theta)
print("The prediction for the 101st day is", predict1)

plt.scatter(X[:,1], y,marker="x")
plt.plot(X[:,1], np.dot(X, theta))
plt.show()

