
# coding: utf-8

# In[100]:

import numpy as np
from matplotlib import pyplot as plt
import time
class Perceptron :

    """An implementation of the perceptron algorithm.
    Note that this implementation does not include a bias term"""

    def __init__(self, max_iterations=150, learning_rate=0.2) :

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
    def fit(self, X, y, X_scaled) :
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
        array_of_eout_scale = []

        self.w = np.random.uniform(-1, 1, len(X[0]))
        temp = self.w
#         print len(X)
#         arrayToI = [1, 2, 4, 8, 32, 64, 128, 256, 512, 1024, 2048, 4096]
        arrayToI = []
        for q in range(169) :
            arrayToI.append(q+1)
#         arrayToI = [170]
#         arrayToI = [1]
#         arrayToI = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1500]
#         arrayToI = []
#         for l in range(499) :p
#             arrayToI.append((l+1)*3)
        print 'Initial w'
        print self.w
        for v in range(len(arrayToI)) :
#             arra_of_ein.append(self.geteot(X[:arrayToI[v]]))
            self.genw(X[:arrayToI[v]])
#             print 'w'
#             print self.w
            array_of_eout.append(self.geteot(X))
        self.w = temp
        print 'Int w'
        print self.w
        for v in range(len(arrayToI)) :
#             arra_of_ein.append(self.geteot(X[:arrayToI[v]]))
            self.genw(X_scaled[:arrayToI[v]])
#             print 'w'
#             print self.w
            array_of_eout_scale.append(self.geteot(X_scaled))
            
#         print 'Final w'
#         print self.w   
        print 'Iteration Used', self.max_iterations
        print 'Learning Rate', self.learning_rate
        print 'Eout Set', array_of_eout
        print 'Eout Set Scale', array_of_eout_scale

#         print 'array In', arra_of_ein
#         print 'array Of I', arrayToI
        plot_LC(arra_of_ein, array_of_eout, array_of_eout_scale, arrayToI)
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
    
    data=np.genfromtxt("heart.data", delimiter=",", comments="#")
#     print data
    
    inputArray = []
    inputArrayScaled = []
    outArray = []
    maxArray = []
    minArray = []
    for i in range(len(data)):    
        innerArray = [1]
        for j in range(len(data[0])): 
            if (j > 1) :
                innerArray.append(data[i][j])
            elif (j  == 1) :
                outArray.append(data[i][j])
        inputArray.append(innerArray)
        
        
    X = np.asarray(inputArray)
    y = np.asarray(outArray)
    maxValue = X.max(axis=0)
    minValue = X.min(axis=0)
    for i in range(len(inputArray)) :
        innerScaleArray = [1]
        for j in range(len(inputArray[0])-1) :
            innerScaleArray.append((((inputArray[i][j+1] - minValue[j+1]) * 1.0 / (maxValue[j+1] - minValue[j+1]) * 1.0)*2) - 1)
        inputArrayScaled.append(innerScaleArray)
    
    X_scaled = np.asarray(inputArrayScaled)
#     X_scaled[:,0] = 1.0
    print 'X'
    print X
    print 'X Scaled'
    print X_scaled
    
    
#     maxValue = np.max(np.absolute(inputArray))
#     minValue = np.min(inputArray)
#     multiplier = 2
#     difference = -1
#     print minValue, maxValue
#     inputArray = (inputArray - minValue) * 1.0 / (maxValue - minValue) * 1.0
#     inputArray = (inputArray * multiplier) + difference
  
#     inputArray[:, 0] = 1
    
    
#     y_scaled = np.asarray(outArray)
    print 'Fetching Time,', time.time() - start_time  
#     print 'new Y', y
#     print inputArray
#     print outArray
#     X = np.random.uniform(-1, 1, [40, 2])
# #     print X,X.shape
#     y = np.sign(np.dot(X, w))
#     print 'new Y', y
    w = np.random.uniform(-1, 1, len(X[0]))
    return X,y,w,X_scaled

def plot_LC(arra_of_ein, array_of_eout, array_of_eout_scale, arrayToI) :
#     print arra_of_ein
#     fig = plt.figure(figsize=(500,500))
#     plt.xlim(0,180)
#     plt.ylim(0.1,1)
# #     ax = plt.subplot(111)

# #     a = -w[0]/w[1]
# #     pts = np.linspace(-100,100)
# #     plt.plot(pts, a*pts, 'k-')
# #     cols = {1: 'r', -1: 'b'}
#     for i in range(len(arrayToI)): 
# #         plt.plot(arrayToI[i], arra_of_ein[i], 'r'+'o')
#         plt.plot(arrayToI[i], array_of_eout[i], 'hello')
# #     plt.plot(arrayToI, arra_of_ein)
#     plt.plot(arrayToI, array_of_eout)
# #     ax
# #     ax.legend()
#     plt.xlabel('Number Of Examples')
#     plt.ylabel('Out-Of-Sample Error')
# #     plt.savefig("Learning.pdf")
#     plt.show()
#     fig.savefig('Learning.pdf')
    
    fig = plt.figure()

    ax = plt.subplot(111)
    plt.xlim(0,180)
    plt.ylim(0.1,1)
#     a = -w[0]/w[1]
#     pts = np.linspace(-100,100)
#     plt.plot(pts, a*pts, 'k-')
#     cols = {1: 'r', -1: 'b'}
    for i in range(len(arrayToI)): 
#         plt.plot(arrayToI[i], arra_of_ein[i], 'r'+'o')
        ax.plot(arrayToI[i], array_of_eout[i])
        ax.plot(arrayToI[i], array_of_eout_scale[i])

#     plt.plot(arrayToI, arra_of_ein)
    ax.plot(arrayToI, array_of_eout, label = 'Out-Of-Sample Error Using Original Dataset')
    ax.plot(arrayToI, array_of_eout_scale, label = 'Out-Of-Sample Error Using Scale Dataset')

#     ax
#     ax.legend()
    plt.xlabel('Number Of Examples')
    plt.ylabel('Out-Of-Sample Error')
#     plt.savefig("Learning.pdf")
    ax.legend()
    plt.show()
#     fig.savefig('Learning.pdf')
    

def plot_data(X, y, w) :
    fig = plt.figure(figsize=(500,500))
    ax = plt.subplot(111)
    plt.xlim(0,90)
    plt.ylim(-20,20)
#     a = -w[0]/w[1]
    pts = np.linspace(-100,100)
    plt.plot(pts, a*pts, 'k-')
    cols = {1: 'r', -1: 'b'}
    for i in range(len(X)): 
        plt.plot(X[i][0], X[i][1], cols[y[i]]+'o')
    plt.savefig("Learning.pdf")
#     plt.show()

if __name__=='__main__' :
    X,y,w,X_scaled = generate_separable_data()
    p = Perceptron()
#     print 'X'
#     print X
#     print 'Y'
#     print y
    p.fit(X,y,X_scaled)


# In[ ]:




# In[ ]:



