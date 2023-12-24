import numpy as np
import time
import scipy.sparse

class SoftmaxRegression(object):

    def __init__(self):

        self.input_size = 784  # input vector size
        self.num_classes = 10  # number of classes
        self.lamda = 0.001  # weight decay parameter

        rand = np.random.RandomState(int(time.time()))

        self.theta = 0.005 * np.asarray(rand.normal(size=(10 * 784, 1)))

    def getGroundTruth(self, labels):

        labels = np.array(labels).flatten()
        data = np.ones(len(labels))
        indptr = np.arange(len(labels) + 1)

        ground_truth = scipy.sparse.csr_matrix((data, labels, indptr))
        ground_truth = np.transpose(ground_truth.todense())

        return ground_truth

    def softmaxCost(self, theta, input, labels):

        ground_truth = self.getGroundTruth(labels)

        theta = theta.reshape(self.num_classes, self.input_size)

        theta_x = np.dot(theta, input)
        hypothesis = np.exp(theta_x)
        probabilities = hypothesis / np.sum(hypothesis, axis=0)

        cost_examples = np.multiply(ground_truth, np.log(probabilities))
        traditional_cost = -(np.sum(cost_examples) / input.shape[1])

        theta_squared = np.multiply(theta, theta)
        weight_decay = 0.5 * self.lamda * np.sum(theta_squared)

        cost = traditional_cost + weight_decay

        theta_grad = -np.dot(ground_truth - probabilities, np.transpose(input))
        theta_grad = theta_grad / input.shape[1] + self.lamda * theta
        theta_grad = np.array(theta_grad)
        theta_grad = theta_grad.flatten()

        return [cost, theta_grad]

    def softmaxPredict(self, input):
        self.theta = self.theta.reshape(self.num_classes, self.input_size)
        theta_x = np.dot(self.theta, input)
        hypothesis = np.exp(theta_x)
        probabilities = hypothesis / np.sum(hypothesis, axis=0)

        predictions = np.argmax(probabilities, axis=0).reshape(-1, 1)

        return predictions
    
    def save_weights(self, file_path):
        np.savez(file_path, theta=self.theta)
    def load_weights(self, file_path):
        data = np.load(file_path)
        self.theta = data['theta']