# Percepton NN Classifier

'''
The algorithm that we’ll be implementing is a Perceptron which is one of the very first machine learning algorithm.
The Perceptron algorithm is simple but powerful. Given a training dataset, the algorithm automatically learns 
“the optimal weight coefficients that are then multiplied with the input features in order to make the decision of whether a neuron fires or not”
'''

'''
# algorithm flow is 

1) Initialize an array with weights equal to 0. The array length is equal to no. of attributes/features plus one. The additional feature is the threshold. 
Note that: For the percepton algo, the features must be numeric value.

self.w_ = np.zeros(1+X.shape[1])

2) We start a loop equal to number of iterations 'num_iterations'. This hyperparameter is decided by data scientist.

for _ in range(self.num_iterations):

3) We start a loop on each training data points & it's target.
   The prediction calculation is a matricial multiplication of the features with their respective weights. To this multiplication we add the value of the threshold. 
   If the result is above 0, the predicted category is 1. If the result is below 0, the predicted category is -1.

 At each iteration on the data point, if the prediction is not accurate, the algorithm will adjust the weights until convergence. 
 Note: the adjustments are made propotionally to the difference between the target & predicted value.This difference is then multiplied by the learning rate eta, which
 is an hyperparameter between 0 & 1. The higher is eta, the larger corrections of weights is required. 
If the prediction is correct, the algorithm won't adjust the weights.

self.w_ = np.zeros(1+X.shape[1])

for _ in range(self.num_iterations):
 for xi, target in zip(X,y):
  update = self.eta * (target - self.predict(xi))
  self.w_[1:] += update * xi
  self.w_[0] += update

def net_input(self,X):
 """ Calculate net input"""
 return np.dot(X,self.w_[1:]) + self.w_[0]

def predict(self, X):
 """Return class label after unit step"""
 return np.where(self.net_input(X)>=0.0, 1, -1)

""" The perceptron will converge only if two classes are linearly separable.
Simply said, if you are able to draw straight line to entirely separate the
two classes, the algorithm will converge. Else, the algorithm will keep 
iterating and will readjust weights until it rreaches the maximum number
of iterations , i.e., num_iterations"""
'''
###############################################################################################################

import numpy as np

class Simple_Classifier_Perceptron(object):
	""" Perceptron classifier

	Parameters
	----------
	eta: float (Learning rate between 0 - 1)
	num_iterations: int (passes over the training dataset)


	Attributes:
	-----------
	w_: 1d-array (Weights after fitting)
	errors_: list (Number of misclassifications (updates) in every epoch)
	"""

	def __init__(self, eta=0.01, num_iterations=50):
		self.eta = eta
		self.num_iterations = num_iterations

	def fit(self, X, y):

		""" Fit training data.

		Parameters:
		-----------
		X: {array-like}, shape=[n_samples, n_features] -------> (predictors)
		y: {array-like}, shape=[n_samples] ------------> (target output)
		"""

		self.w_ = np.zeros(1+X.shape[1])
		self.errors_ = list()

		for _ in range(self.num_iterations):
			errors = 0
			for xi, target in zip(X,y):
				update = self.eta * (target-self.predict(xi))
				self.w_[1:] += update * xi
				self.w_[0] += update
				errors += int(update!=0.0)
			self.errors_.append(errors)
		return self


	def net_input(self, X):
		"""Calculate net input"""
		return np.dot(X, self.w_[1:]) + self.w_[0]


	def predict(self,X):
		"""Return class label after unit step"""
		return np.where(self.net_input(X) >= 0.0,1,-1)  # 1 -1    # 2, 1    # 1, 2

###############################################################################################################

