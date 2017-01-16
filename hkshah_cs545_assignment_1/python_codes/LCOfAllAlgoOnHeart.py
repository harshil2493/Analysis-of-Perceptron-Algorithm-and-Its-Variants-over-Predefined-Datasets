
# coding: utf-8

# In[5]:

import numpy as np
from matplotlib import pyplot as plt
import time
class Perceptron :

    """An implementation of the perceptron algorithm.
    Note that this implementation does not include a bias term"""

    def __init__(self, max_iterations=50, learning_rate=0.2, constantC = 0.01) :

        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.constantC = constantC
        
    def geteout(self, Y) :
        count = 0
        for i in range(len(Y)) :
                if y[i] * self.discriminant(Y[i]) <= 0 :
                    count += 1
        finalV = (count * 1.0) / (len(Y) * 1.0)
        return finalV
    def genModified(self, Z) :
#         self.w = np.random.uniform(-1, 1, len(X[0]))
        converged = False
        discriminant = True
        iterations = 0
        while (not converged and iterations < self.max_iterations) :
            converged = True
            
            discriminant = False
            for i in range(len(Z)) :
                neededI = 0
                maxLembda = y[i] * self.discriminant(Z[i])
                if y[i] * self.discriminant(Z[i]) < self.constantC * np.sqrt(np.dot(self.w, self.w)) :
                    
                    for j in range(len(Z)):
#                     print 'LembdaI', y[i] * self.discriminant(X[i]) 
                        tempW = self.w + y[j] * self.learning_rate * Z[j]
                        tempLambda = y[i] * np.dot(tempW, Z[i])
        
                        if maxLembda < tempLambda :
                            maxLembda = tempLambda
                            neededI = j
                    self.w = self.w + y[neededI] * self.learning_rate * Z[neededI]
                    discriminant = True
                    converged = False
#                     print 'print', y[i] * self.discriminant(X[i]
                   

            if(not discriminant) :
                converged = True
            else :
#                 element = neededI
#                 self.w = self.w + y[element] * self.learning_rate * X[element]
                converged = False
            iterations += 1
#             print 'done', iterations
        self.converged = converged
        self.wForM = self.w

    def genwPocket(self, Z) :
        
        
        
        self.wPocket = np.zeros(len(Z[0]))
        misclassifiedCount = 0
        misclassifiedPocketCount = len(Z)
        converged = False
        iterations = 0
        while (not converged and iterations < self.max_iterations) :
            converged = True
            for i in range(len(Z)) :
                if y[i] * self.discriminant(Z[i]) <= 0 :
#                     print '-changing w-'
                    self.w = self.w + y[i] * self.learning_rate * Z[i]                    
                    converged = False
#                     misclassifiedCount = 0                    
#                     plot_data(X, y, self.w)
            iterations += 1
            misclassifiedCount = 0
            for j in range(len(Z)) :
                if y[j] * self.discriminant(Z[j]) <= 0 :
                    misclassifiedCount = misclassifiedCount + 1
            if misclassifiedCount < misclassifiedPocketCount :
                misclassifiedPocketCount = misclassifiedCount
#                         print '-changing wPocket-'
                self.wPocket = np.copy(self.w)
        self.w = self.wPocket
        self.converged = converged
        self.wForPocket = self.w
#         self.w = self.wPocket
    def genw(self, Z) :
        converged = False
        iterations = 0
#         self.w = []
#         self.w = np.zeros(len(Z[0]))

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
        iterations = 0
        if(len(Z[0]) == 13) : 
            self.wForUB = self.w
        else :
            self.wForB = self.w
#         print 'Z Dimentions', len(Z[0])
#         print 'Bias/UnBias', self.w
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
    def fit(self, X, XforOthers, y) :
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
        array_of_eout_withoutbias = []
        array_of_eout_pocket = []
        array_of_eout_modified = []
        arra_of_ein = []
#         self.w = np.random.uniform(-1, 1, len(X[0]))
#         print len(X)
        arrayToI = [1, 2, 4, 8, 32, 64, 128, 170]
#         arrayToI = [1]
#         arrayToI = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1500]
#         arrayToI = []
#         for l in range(499) :p
#             arrayToI.append((l+1)*3)
                
        wForUB = np.random.uniform(-1, 1, len(XforOthers[0]))
        wForB = np.random.uniform(-1, 1, len(X[0]))

        wForPocket = np.random.uniform(-1, 1, len(XforOthers[0]))
        wForM = np.random.uniform(-1, 1, len(XforOthers[0]))
        for v in range(len(arrayToI)) :
#             arra_of_ein.append(self.geteout(X[:arrayToI[v]]))
            self.w = []
            self.w = wForB
            self.genw(X[:arrayToI[v]])
            array_of_eout.append(self.geteout(X))
            self.w = []
            self.w = wForUB
            self.genw(XforOthers[:arrayToI[v]])
            array_of_eout_withoutbias.append(self.geteout(XforOthers))
            self.w = []
            self.w = wForPocket
            self.genwPocket(XforOthers[:arrayToI[v]])
            array_of_eout_pocket.append(self.geteout(XforOthers))
            self.w = []
            self.w = wForM
#             print 'updated w', self.w
            self.genModified(XforOthers[:arrayToI[v]])
            array_of_eout_modified.append(self.geteout(XforOthers))

#         print 'Final w'
#         print self.w    
#         print 'Eout Set', array_of_eout
#         print 'array In', arra_of_ein
#         print 'array Of I', arrayToI
        plot_LC(arra_of_ein, array_of_eout, array_of_eout_withoutbias, array_of_eout_pocket, array_of_eout_modified, arrayToI)
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
    outArray = []
    inputArrayForOthers = []
    for i in range(len(data)):    
        innerArray = [1]
        innerArrayForOthers = []
        for j in range(len(data[0])):
            if j > 1 :
                innerArray.append(data[i][j])
                innerArrayForOthers.append(data[i][j])
            elif j == 1 :
                outArray.append(data[i][j])
        inputArray.append(innerArray)
        inputArrayForOthers.append(innerArrayForOthers)
    X = np.asarray(inputArray)
    XforOthers = np.asarray(inputArrayForOthers)
    y = np.asarray(outArray)
#     y = np.genfromtxt("gisette_train.labels")
    print 'Fetching Time,', time.time() - start_time  
#     print 'new Y', y
#     print inputArray
#     print outArray
#     X = np.random.uniform(-1, 1, [40, 2])
# #     print X,X.shape
#     y = np.sign(np.dot(X, w))
#     print 'new Y', y
    w = np.random.uniform(-1, 1, len(X[0]))
    return X,y,w,XforOthers

def plot_LC(arra_of_ein, array_of_eout, array_of_eout_withoutbias, array_of_eout_pocket, array_of_eout_modified, arrayToI) :
#     print arra_of_ein
#     fig = plt.figure(figsize=(500,500))
    fig = plt.figure()

    ax = plt.subplot(111)
    plt.xlim(0,170)
    plt.ylim(0.2,0.7)
#     a = -w[0]/w[1]
#     pts = np.linspace(-100,100)
#     plt.plot(pts, a*pts, 'k-')
#     cols = {1: 'r', -1: 'b'}
#     for i in range(len(arrayToI)): 
# #         plt.plot(arrayToI[i], arra_of_ein[i], 'r'+'o')
#         plt.plot(arrayToI[i], array_of_eout[i], 'b'+'o')
#         plt.plot(arrayToI[i], array_of_eout_withoutbias[i], 'r'+'o')
#         plt.plot(arrayToI[i], array_of_eout_pocket[i], 'k'+'o')
#         plt.plot(arrayToI[i], array_of_eout_modified[i], 'm'+'o')
        
#     print 'Pocket Eout', array_of_eout_pocket
#     print 'Bias', array_of_eout
#     print 'WithOut Bias', array_of_eout_withoutbias
#     plt.plot(arrayToI, arra_of_ein)
    ax.plot(arrayToI, array_of_eout, label = 'PLA With Bias Term')
    ax.plot(arrayToI, array_of_eout_withoutbias, label = 'PLA Without Bias Term')
    ax.plot(arrayToI, array_of_eout_pocket, label = 'Pocket Algorithm')
    ax.plot(arrayToI, array_of_eout_modified, label = 'Modified PLA')
    ax.legend()
#     ax
    plt.xlabel('Number Of Examples')
    plt.ylabel('Out-Of-Sample Error')
#     plt.savefig("Learning.pdf")
    plt.show()
#     fig.savefig('Learning.pdf')

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
    X,y,w,XforOthers = generate_separable_data()
    p = Perceptron()
#     print 'X'
#     print X
#     print 'Y'
#     print y
    p.fit(X, XforOthers, y)


# In[ ]:



