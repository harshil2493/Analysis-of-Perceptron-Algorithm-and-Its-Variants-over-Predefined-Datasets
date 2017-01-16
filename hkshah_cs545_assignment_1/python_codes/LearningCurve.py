
# coding: utf-8

# In[1]:

import numpy as np
from matplotlib import pyplot as plt
import time
class Perceptron :

    """An implementation of the perceptron algorithm.
    Note that this implementation does not include a bias term"""

    def __init__(self, max_iterations=2, learning_rate=0.2) :

        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        
    def geteot(self, Y) :
        count = 0
        for i in range(len(Y)) :
                if y[i] * self.discriminant(Y[i]) <= 0 :
                    count += 1
        finalV = (count * 1.0) / (len(Y) * 1.0)
        return finalV
    def genw(self, Z) :
        converged = False
        iterations = 0
        while (not converged and iterations < self.max_iterations) :
#         while (not converged) :

            converged = True
            for i in range(len(Z)) :
                if y[i] * self.discriminant(Z[i]) <= 0 :
#                     print 'data', y[i] * self.learning_rate * X[i]
                    self.w = self.w + y[i] * self.learning_rate * Z[i]
#                     print self.w
#                     print 'temp', temporary
#                     for k in range(len(self.w)):
#                         self.w[k] = self.w[k] + temporary[k]
#                     self.w = self.w + temporary
                    converged = False
            iterations += 1
#             print  iterations
#         self.converged = converged
#         if converged or iterations == self.max_iterations:
#             print 'w', self.w
#             miniofarray = np.min(self.w)
            
# #             print miniofarray
#             for j in range(len(self.w)) :
# #                 print self.w[j]
#                 self.w[j] = self.w[j]/miniofarray
# #                 print self.w[j]
            
#             print 'w'
#             print '-'
#             print self.w
#             print 'learning rate ', self.learning_rate
#             print 'converged in %d iterations ' % iterations
    def fit(self, X, y) :
        """
        Train a classifier using the perceptron training algorithm.
        After training the attribute 'w' will contain the perceptron weight vector.

        Parameters
        ----------

        X : ndarray, shape (num_examples, n_features)
        Training data.

        y : ndarray, shape (n_examples,)
        Array of labels.
        
        """
        array_of_eout = []
        arra_of_ein = []
        self.w = np.random.uniform(-1, 1, len(X[0]))
#         print len(X)
        arrayToI = [5, 10, 20, 40, 80, 160, 320, 640, 1280, 2560, 4500]
#         arrayToI = [1, 2, 4, 8, 32, 64, 128, 256, 512, 1024, 2048, 4096]
#         arrayToI = [1]
#         arrayToI = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1500]
#         arrayToI = []
#         for l in range(499) :p
#             arrayToI.append((l+1)*3)
        print 'Initial w'
        print self.w
        for v in range(len(arrayToI)) :
            
            self.genw(X[:arrayToI[v]])
            arra_of_ein.append(self.geteot(X[:arrayToI[v]]))
            array_of_eout.append(self.geteot(X))
            

        print 'Final w'
        print self.w    
        print 'Eout Set', array_of_eout
#         print 'array In', arra_of_ein
#         print 'array Of I', arrayToI
        plot_LC(arra_of_ein, array_of_eout, arrayToI)
#         array_of_eout.append(self.geteot(X))
#         arra_of_ein.append(self.geteot(X[:20]))
        
#         self.genw(X[:20])
        
#         print self.w
#         converged = False
#         iterations = 0
#         while (not converged and iterations < self.max_iterations) :
# #         while (not converged) :

#             converged = True
#             for i in range(len(X)) :
#                 if y[i] * self.discriminant(X[i]) <= 0 :
# #                     print 'data', y[i] * self.learning_rate * X[i]
#                     self.w = self.w + y[i] * self.learning_rate * X[i]
# #                     print self.w
# #                     print 'temp', temporary
# #                     for k in range(len(self.w)):
# #                         self.w[k] = self.w[k] + temporary[k]
# #                     self.w = self.w + temporary
#                     converged = False
#             iterations += 1
# #             print  iterations
#         self.converged = converged
#         if converged or iterations == self.max_iterations:
# #             print 'w', self.w
# #             miniofarray = np.min(self.w)
            
# # #             print miniofarray
# #             for j in range(len(self.w)) :
# # #                 print self.w[j]
# #                 self.w[j] = self.w[j]/miniofarray
# # #                 print self.w[j]
            
#             print 'w'
#             print '-'
#             print self.w
#             print 'learning rate ', self.learning_rate
#             print 'converged in %d iterations ' % iterations
# #             print 'minimum of array ', miniofarray
#             plot_data(X, y, self.w)


    def discriminant(self, x) :
#         print np.dot(self.w, x)
        return np.dot(self.w, x)
            
    def predict(self, X) :
        """
        make predictions using a trained linear classifier

        Parameters
        ----------

        X : ndarray, shape (num_examples, n_features)
        Training data.
        """
        
        scores = np.dot(self.w, X)
        return np.sign(scores)

def generate_separable_data() :
    start_time = time.time()
#     xA,yA,xB,yB = [np.random.uniform(-1, 1) for i in range(4)]
   
#     print 'w'
#     print w
#     print w,w.shape
    
    data=np.genfromtxt("gisette_train.data")
#     print data
    
    inputArray = []
    outArray = []
    for i in range(len(data)):    
        innerArray = [1]
        for j in range(len(data[i])):        
            innerArray.append(data[i][j])
           
        inputArray.append(innerArray)
    
    X = np.asarray(inputArray)
    y = np.genfromtxt("gisette_train.labels")
    print 'Fetching Time,', time.time() - start_time  
#     print 'new Y', y
#     print inputArray
#     print outArray
#     X = np.random.uniform(-1, 1, [40, 2])
# #     print X,X.shape
#     y = np.sign(np.dot(X, w))
#     print 'new Y', y
    w = np.random.uniform(-1, 1, len(X[0]))
    return X,y,w

def plot_LC(arra_of_ein, array_of_eout, arrayToI) :
#     print arra_of_ein
    fig = plt.figure()
    ax = plt.subplot(111)
    plt.xlim(-100,4510)
    plt.ylim(-0.1,0.55)
#     a = -w[0]/w[1]
#     pts = np.linspace(-100,100)
#     plt.plot(pts, a*pts, 'k-')
#     cols = {1: 'r', -1: 'b'}
    for i in range(len(arrayToI)): 
        ax.plot(arrayToI[i], arra_of_ein[i])
        ax.plot(arrayToI[i], array_of_eout[i])
    ax.plot(arrayToI, arra_of_ein, label = 'In-Sample Error Using PLA With Biased Algorithm')
    ax.plot(arrayToI, array_of_eout, label = 'Out-Of-Sample Error Using PLA With Biased Algorithm')
#     ax
    plt.xlabel('Number Of Examples')
    plt.ylabel('Error')
    ax.legend()
#     plt.savefig("Learning.pdf")
    plt.show()
    fig.savefig('Learning.pdf')

def plot_data(X, y, w) :
    fig = plt.figure(figsize=(500,500))
    plt.xlim(0,90)
    plt.ylim(-20,20)
    a = -w[0]/w[1]
    pts = np.linspace(-100,100)
    plt.plot(pts, a*pts, 'k-')
    cols = {1: 'r', -1: 'b'}
    for i in range(len(X)): 
        plt.plot(X[i][0], X[i][1], cols[y[i]]+'o')
    plt.savefig("Learning.pdf")
    plt.show()

if __name__=='__main__' :
    X,y,w = generate_separable_data()
    p = Perceptron()
#     print 'X'
#     print X
#     print 'Y'
#     print y
    p.fit(X,y)


# In[ ]:



