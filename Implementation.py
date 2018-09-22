# Implementation of Perceptron and other variants.
import matplotlib.pyplot as plt
import numpy as np

# Generates linearly seperable data in 2-D

def generate_separable_data(N) :
    w = np.random.uniform(-1, 1, 2)
    print(w)
    X = np.random.uniform(-1, 1, [N, 2])
    print (X.shape)
    y = np.sign(np.inner(w, X))
    return X,y,w


class Perceptron :
 
    """An implementation of the perceptron algorithm.
    Note that this implementation does not include a bias term"""
 
    def __init__(self, max_iterations=100, learning_rate=0.2) :
 
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
        converged = False
        iterations = 0
        while (not converged and iterations <= self.max_iterations) :
            converged = True
            for i in range(len(X)) :
                if y[i] * self.decision_function(X[i]) <= 0 :
                    self.w = self.w + y[i] * self.learning_rate * X[i]
                    converged = False
            iterations += 1
        self.converged = converged
        if converged :
            print ('converged in %d iterations ' % iterations)
        print ('weight vector: ' + str(self.w))
 
    def decision_function(self, x) :
        return np.inner(self.w, x)
 
    def predict(self, X) :
        """
        make predictions using a trained linear classifier
 
        Parameters
        ----------
 
        X : ndarray, shape (num_examples, n_features)
        Training data.
        """
 
        scores = np.inner(self.w, X)
        return np.sign(scores)

class Pocket :
 
	"""An implementation of the Pocket algorithm.
	Note that this implementation does not include a bias term"""

	def __init__(self, max_iterations=100, learning_rate=0.2) :

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
	    self.w_pocket = np.zeros(len(X[0]))
	    converged = False
	    iterations = 0
	    while (not converged and iterations <= self.max_iterations) :
	        converged = True
	        for i in range(len(X)) :
	            if y[i] * self.decision_function(X[i]) <= 0 :
	                self.w = self.w + y[i] * self.learning_rate * X[i]

	                # Calculate Ein with Wpocket
	                EinPocket = self.inSample(X, y, self.w_pocket)

	                # Calculate Ein with W
	                Ein = self.inSample(X, y, self.w)

	                # If EinW lesser than EinPocket, then Wpocket = W
	                if Ein < EinPocket:
	                	self.w_pocket = self.w
	                converged = False
	        iterations += 1
	    self.converged = converged
	    if converged :
	        print ('converged in %d iterations ' % iterations)
	    print ('Pocketed weight vector: ' + str(self.w_pocket))
	    #print(str(self.w.shape))

	def inSample(self, X, y, w):
		misclass = 0
		for i in range(len(X)):
			if y[i] * np.inner(w, X[i]) <= 0:
				misclass += 1
		Ein = misclass / len(X)
		return Ein


	def decision_function(self, x) :
	    return np.inner(self.w, x)

	def predict(self, X) :
	    """
	    make predictions using a trained linear classifier

	    Parameters
	    ----------

	    X : ndarray, shape (num_examples, n_features)
	    Training data.
	    """

	    scores = np.inner(self.w, X)
	    return np.sign(scores)

class Adatron :
 
	"""An implementation of the perceptron algorithm.
	Note that this implementation does not include a bias term"""

	def __init__(self, max_iterations=100, learning_rate=0.2) :

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
	    alpha = np.ones(len(X))
	    # Calculate weight
	    self.w = np.zeros(len(X[0]))
	    for i in range(len(X)):
	    	self.w = self.w + alpha[i] * y[i] * X[i]

	    iterations = 0
	    while (iterations <= self.max_iterations) :
	        for i in range(len(X)) :
        	    oldalpha = alpha
	            gamma = y[i] * np.inner(self.w, X[i])
	            da = self.learning_rate * (1 - gamma)
	            if (alpha[i] + da) < 0:
	                alpha[i] = 0
	            else:
	                alpha[i] = alpha[i] + da
	            
	            # UPDATE WEIGHT
	            oldw = self.w
	            neww = (oldw - (oldalpha[i] * y[i] * X[i])) + alpha[i] * y[i] * X[i]
	            self.w = neww
	        
	        iterations += 1

	    print ('converged in %d iterations ' % iterations)
	    print ('weight vector: ' + str(self.w))
	    print('error: ' + str(self.inSample(X, y, self.w)))

	def decision_function(self, x) :
	    return np.inner(self.w, x)

	def inSample(self, X, y, w):
		misclass = 0
		for i in range(len(X)):
			if y[i] * np.inner(w, X[i]) <= 0:
				misclass += 1
		Ein = misclass / len(X)
		return Ein

	def predict(self, X) :
	    """
	    make predictions using a trained linear classifier

	    Parameters
	    ----------

	    X : ndarray, shape (num_examples, n_features)
	    Training data.
	    """

	    scores = np.inner(self.w, X)
	    return np.sign(scores)


if __name__ == '__main__':
	
	X,y,w = generate_separable_data(40)
	
	train_data = np.genfromtxt("gisette_train.data", delimiter = " ")
	valid_data = np.genfromtxt("gisette_valid.data", delimiter = " ")
	#X = np.concatenate((train_data, valid_data), axis = 0)
	train_labels = np.genfromtxt("gisette_train.labels", delimiter = " ")
	valid_labels = np.genfromtxt("gisette_valid.labels", delimiter = " ")
	#y = np.concatenate((train_labels, valid_labels), axis = 0)
	#p1 = Perceptron()
	#p1.fit(X, y)

	p = Adatron()
	p.fit(X, y)
