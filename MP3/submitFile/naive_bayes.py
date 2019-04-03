import numpy as np
import math

class NaiveBayes(object):
	def __init__(self,num_class,feature_dim,num_value):
		"""Initialize a naive bayes model. 

		This function will initialize prior and likelihood, where 
		prior is P(class) with a dimension of (# of class,)
			that estimates the empirical frequencies of different classes in the training set.
		likelihood is P(F_i = f | class) with a dimension of 
			(# of features/pixels per image, # of possible values per pixel, # of class),
			that computes the probability of every pixel location i being value f for every class label.  

		Args:
		    num_class(int): number of classes to classify
		    feature_dim(int): feature dimension for each example 
		    num_value(int): number of possible values for each pixel 
		"""

		self.num_value = num_value
		self.num_class = num_class
		self.feature_dim = feature_dim


		self.prior = np.zeros((num_class))
		self.likelihood = np.zeros((feature_dim,num_value,num_class))

	def train(self,train_set,train_label):
		""" Train naive bayes model (self.prior and self.likelihood) with training dataset. 
			self.prior(numpy.ndarray): training set class prior (in log) with a dimension of (# of class,),
			self.likelihood(numpy.ndarray): traing set likelihood (in log) with a dimension of 
				(# of features/pixels per image, # of possible values per pixel, # of class).
			You should apply Laplace smoothing to compute the likelihood. 

		Args:
		    train_set(numpy.ndarray): training examples with a dimension of (# of examples, feature_dim)
		    train_label(numpy.ndarray): training labels with a dimension of (# of examples, )
		"""
		# YOUR CODE HERE
		k_value=0.1
		num_examples=int(train_set.size/self.feature_dim)
		for each in train_label:
			self.prior[each] += 1/num_examples
		graph_likelihood = np.zeros((num_examples,self.feature_dim))
		new_train_set = train_set.reshape((num_examples,self.feature_dim))
		for i in range(new_train_set.shape[0]):
			for j in range(new_train_set.shape[1]):
				self.likelihood[j][new_train_set[i][j]][train_label[i]] += 1
		for i in range(self.likelihood.shape[0]):
			for j in range(self.likelihood.shape[1]):
				for k in range(self.likelihood.shape[2]):
					self.likelihood[i][j][k] = (self.likelihood[i][j][k]+k_value)/(num_examples*self.prior[train_label[i]]+k_value*self.num_value)



	def test(self,test_set,test_label):
		""" Test the trained naive bayes model (self.prior and self.likelihood) on testing dataset,
			by performing maximum a posteriori (MAP) classification.  
			The accuracy is computed as the average of correctness 
			by comparing between predicted label and true label. 

		Args:
		    test_set(numpy.ndarray): testing examples with a dimension of (# of examples, feature_dim)
		    test_label(numpy.ndarray): testing labels with a dimension of (# of examples, )

		Returns:
			accuracy(float): average accuracy value  
			pred_label(numpy.ndarray): predicted labels with a dimension of (# of examples, )
		"""    

		# YOUR CODE HERE
		num_examples1=int(test_set.size/self.feature_dim)
		new_test_set = test_set.reshape((num_examples1,self.feature_dim))
		accuracy = 0.0
		pred_label = np.zeros((len(test_set)))
		class_map = np.zeros((self.num_class))
		for i in range(new_test_set.shape[0]):
			for k in range(self.num_class):
				_map = math.log(self.prior[k],10)
				for j in range(new_test_set.shape[1]):
					_map = _map+math.log(self.likelihood[j][new_test_set[i][j]][k],10)
				class_map[k] = _map
			pred_label[i] = np.argmax(class_map)


		for i in range(pred_label.size):
			if pred_label[i] == test_label[i]:
				accuracy += 1
		accuracy = accuracy/num_examples1
		print(accuracy)
		return accuracy, pred_label


	def save_model(self, prior, likelihood):
		""" Save the trained model parameters 
		"""    

		np.save(prior, self.prior)
		np.save(likelihood, self.likelihood)

	def load_model(self, prior, likelihood):
		""" Load the trained model parameters 
		""" 

		self.prior = np.load(prior)
		self.likelihood = np.load(likelihood)

	def intensity_feature_likelihoods(self, likelihood):
		# """
	    # Get the feature likelihoods for high intensity pixels for each of the classes,
	    #     by sum the probabilities of the top 128 intensities at each pixel location,
	    #     sum k<-128:255 P(F_i = k | c).
	    #     This helps generate visualization of trained likelihood images.
	    #
	    # Args:
	    #     likelihood(numpy.ndarray): likelihood (in log) with a dimension of
	    #         (# of features/pixels per image, # of possible values per pixel, # of class)
	    # Returns:
	    #     feature_likelihoods(numpy.ndarray): feature likelihoods for each class with a dimension of
	    #         (# of features/pixels per image, # of class)
	    # """
		feature_likelihoods = np.zeros((likelihood.shape[0],likelihood.shape[2]))
		for i in range(likelihood.shape[0]):
			for j in range(128,likelihood.shape[1]):
				for k in range(likelihood.shape[2]):
					#if likelihood[i][j][k] >=128 and likelihood[i][j][k]<=255:
					feature_likelihoods[i][k] += likelihood[i][j][k]
		return feature_likelihoods
