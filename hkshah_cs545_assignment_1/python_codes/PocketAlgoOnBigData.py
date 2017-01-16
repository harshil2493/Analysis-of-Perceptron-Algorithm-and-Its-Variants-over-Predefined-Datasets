
# coding: utf-8

# In[2]:

import numpy as np
from matplotlib import pyplot as plt
import time
class Perceptron :

    """An implementation of the perceptron algorithm.
    Note that this implementation does not include a bias term"""
    def geteot(self, Y) :
        count = 0
        for i in range(len(Y)) :
                if y[i] * self.discriminant(Y[i]) <= 0 :
                    count += 1
        finalE = (count * 1.0) / (len(Y) * 1.0)
        return finalE
    def geteout(self) :
        datafinal=np.genfromtxt("gisette_test.data")
#     print data
    
        inputArray = []
        outArray = []
        for i in range(len(datafinal)):    
            innerArray = []
            for j in range(len(datafinal[i])):        
                innerArray.append(datafinal[i][j])
           
            inputArray.append(innerArray)
    
        Xfinal = np.asarray(inputArray)
        yfinal = np.genfromtxt("gisette_test.labels")
        
        count = 0
        for i in range(len(Xfinal)) :
            if yfinal[i] * self.discriminant(Xfinal[i]) <= 0 :
                count += 1
        finalEout = (count * 1.0) / (len(Xfinal) * 1.0)
        return finalEout
    def __init__(self, max_iterations=200, learning_rate=0.1) :

        self.max_iterations = max_iterations
        self.learning_rate = learning_rate

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
        self.w = np.zeros(len(X[0]))
        
        print 'w'
        print '-'
        print 'initialized to', self.w
        
        self.wPocket = np.zeros(len(X[0]))
        misclassifiedCount = 0
        misclassifiedPocketCount = len(X)
        converged = False
        iterations = 0
        while (not converged and iterations < self.max_iterations) :
            converged = True
            for i in range(len(X)) :
                if y[i] * self.discriminant(X[i]) <= 0 :
#                     print '-changing w-'
                    self.w = self.w + y[i] * self.learning_rate * X[i]                    
                    converged = False
#                     misclassifiedCount = 0                    
#                     plot_data(X, y, self.w)
            iterations += 1
            misclassifiedCount = 0
            for j in range(len(X)) :
                if y[j] * self.discriminant(X[j]) <= 0 :
                    misclassifiedCount = misclassifiedCount + 1
            if misclassifiedCount < misclassifiedPocketCount :
                misclassifiedPocketCount = misclassifiedCount
#                         print '-changing wPocket-'
                self.wPocket = np.copy(self.w)
        self.w = self.wPocket
        self.converged = converged
#         if converged :
#             print 'converged in %d iterations ' % iterations
#         print 'converged in %d iterations ' % iterations
#         print '-selfw-', self.wPocket
#         print '-w-', self.w
#         plot_data(X,y,self.wPocket)
#         plot_data(X,y,self.w)
        if converged or iterations == self.max_iterations:
#             print 'w', self.w
#             miniofarray = np.min(self.w)
            
# #             print miniofarray
#             for j in range(len(self.w)) :
# #                 print self.w[j]
#                 self.w[j] = self.w[j]/miniofarray
#                 print self.w[j]
            
            print 'w'
            print '-'
            print self.w
            print 'learning rate ', self.learning_rate
            print 'converged in %d iterations ' % iterations
            print 'Ein = ',self.geteot(X)
            print 'Eout = ', self.geteout()

#             print 'minimum of array ', miniofarray
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
        innerArray = []
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



