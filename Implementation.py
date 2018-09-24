# Implementation of Perceptron and other variants.
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table, Column

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
 
    def __init__(self, max_iterations=100, learning_rate=0.2, bias = 0) :
 
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.bias = bias
 
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

        # ADDING BIAS TERM

        # Change input to accomodate bias. Tack a column of 1s.
        X = np.insert(X, 0, 1, 1)
        # Include bias in weight
        self.w = np.insert(self.w, 0, self.bias)

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

	def __init__(self, max_iterations=100, learning_rate=0.2, bias = 0) :

	    self.max_iterations = max_iterations
	    self.learning_rate = learning_rate
	    self.bias = bias

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

	    # ADDING BIAS TERM

        # Change input to accomodate bias. Tack a column of 1s.
	    X = np.insert(X, 0, 1, 1)

        # Include bias in weights
	    self.w = np.insert(self.w, 0, self.bias)
	    self.w_pocket = np.insert(self.w_pocket, 0, self.bias)

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

	def __init__(self, max_iterations=100, learning_rate=0.2, bias = 0) :

	    self.max_iterations = max_iterations
	    self.learning_rate = learning_rate
	    self.bias = bias

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

     	# ADDING BIAS TERM

        # Change input to accomodate bias. Tack a column of 1s.
	    X = np.insert(X, 0, 1, 1)
        # Include bias in weight
	    self.w = np.insert(self.w, 0, self.bias)

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
	print(X)
	print(y)
	train_data = np.genfromtxt("gisette_train.data", delimiter = " ")
	valid_data = np.genfromtxt("gisette_valid.data", delimiter = " ")
	X = np.concatenate((train_data, valid_data), axis = 0)
	train_labels = np.genfromtxt("gisette_train.labels", delimiter = " ")
	valid_labels = np.genfromtxt("gisette_valid.labels", delimiter = " ")
	y = np.concatenate((train_labels, valid_labels), axis = 0)
	p1 = Perceptron(bias = 10)
	p1.fit(X, y)

	#p = Pocket(bias = 10)
	#p.fit(X, y)
	

	# QSAR DATASET
	#data = np.genfromtxt("biodeg.csv", delimiter = ',')
	#print(type(data))
	#print(data[0, :])
	#print(data.shape)
	
	#REMOVE NANS
	#data = data[~np.isnan(data).any(axis=1)]
	#print(data.shape)
	#FIRST STEP IS TO EXTRACT THE LAST COLUMN
	#outdata = data[:,-1]# np.genfromtxt("processed.cleveland.data", delimiter = ",", usecols = -1)

	# PROCESS INTO -1 AND 1
	# EXPECTED OUTPUTS
	#outdata[outdata > 0] = -1
	#outdata[outdata == 0] = 1
	#print(outdata.shape)

	# INPUTS
	#indata = np.delete(data, -1, 1)

	#data = np.genfromtxt("cleveland.data", delimiter = ",")
	
