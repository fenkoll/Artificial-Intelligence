import numpy as np

class MultiClassPerceptron(object):
	def __init__(self,num_class,feature_dim):
		"""Initialize a multi class perceptron model. 

		This function will initialize a feature_dim weight vector,
		for each class. 

		The LAST index of feature_dim is assumed to be the bias term,
			self.w[:,0] = [w1,w2,w3...,BIAS] 
			where wi corresponds to each feature dimension,
			0 corresponds to class 0.  

		Args:
		    num_class(int): number of classes to classify
		    feature_dim(int): feature dimension for each example 
		"""

		self.w = np.zeros((feature_dim+1,num_class))

	def train(self,train_set,train_label):
		""" Train perceptron model (self.w) with training dataset. 

		Args:
		    train_set(numpy.ndarray): training examples with a dimension of (# of examples, feature_dim)
		    train_label(numpy.ndarray): training labels with a dimension of (# of examples, )
		"""

		# YOUR CODE HERE\
		image_type = 0
		num_examples=int(train_set.size/(self.w.shape[0]-1))
		w = np.transpose(self.w)
		new_train_set = train_set.reshape(num_examples,self.w.shape[0]-1)
		ARRAY = np.ones(num_examples)
		new_train_set = np.insert(new_train_set,w.shape[1]-1,values=ARRAY, axis=1)
		for i in range(new_train_set.shape[0]):
			score = 0
			for j in range(w.shape[0]):
				new_score = 0
				new_score_product = np.multiply(w[j],new_train_set[i])
				new_score = np.sum(new_score_product)
				if new_score > score:
					score = new_score
					image_type = j
			if image_type!=train_label[i]:
				for p in range(w.shape[1]):
					w[image_type][p] -= new_train_set[i][p]
					w[train_label[i]][p] += new_train_set[i][p]

	def test(self,test_set,test_label):
		""" Test the trained perceptron model (self.w) using testing dataset. 
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
		accuracy = 0
		pred_label = np.zeros((len(test_set)))
		num_examples1=int(test_set.size/(self.w.shape[0]-1))
		new_test_set = test_set.reshape((num_examples1,self.w.shape[0]-1))
		bias = np.ones(num_examples1)
		w = np.transpose(self.w)
		new_test_set = np.insert(new_test_set,w.shape[1]-1,values=bias,axis=1)
		for i in range(new_test_set.shape[0]):
			score = 0
			for j in range(w.shape[0]):
				new_score = 0
				new_score_product = np.multiply(w[j],new_test_set[i])
				new_score = np.sum(new_score_product)
				if new_score>score:
					image_type = j
					score = new_score
			pred_label[i] = image_type

		for i in range(pred_label.size):
			if pred_label[i] == test_label[i]:
				accuracy += 1/num_examples1
		print(accuracy)
		return accuracy, pred_label

	def save_model(self, weight_file):
		""" Save the trained model parameters 
		""" 

		np.save(weight_file,self.w)

	def load_model(self, weight_file):
		""" Load the trained model parameters 
		""" 

		self.w = np.load(weight_file)

