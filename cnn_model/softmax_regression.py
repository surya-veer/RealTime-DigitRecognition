import struct
import numpy as np
import array
import time
import scipy.sparse
import scipy.optimize

class SoftmaxRegression(object):

    def __init__(self, input_size, num_classes, lamda):

        self.input_size = input_size  # input vector size
        self.num_classes = num_classes  # number of classes
        self.lamda = lamda  # weight decay parameter

        rand = np.random.RandomState(int(time.time()))

        self.theta = 0.005 * np.asarray(rand.normal(size=(num_classes * input_size, 1)))

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

    def softmaxPredict(self, theta, input):

        theta = theta.reshape(self.num_classes, self.input_size)

        theta_x = np.dot(theta, input)
        hypothesis = np.exp(theta_x)
        probabilities = hypothesis / np.sum(hypothesis, axis=0)

        predictions = np.zeros((input.shape[1], 1))
        predictions[:, 0] = np.argmax(probabilities, axis=0)

        return predictions

def loadMNISTImages(file_name):

    image_file = open(file_name, 'rb')

    head1 = image_file.read(4)
    head2 = image_file.read(4)
    head3 = image_file.read(4)
    head4 = image_file.read(4)

    num_examples = struct.unpack('>I', head2)[0]
    num_rows = struct.unpack('>I', head3)[0]
    num_cols = struct.unpack('>I', head4)[0]

    dataset = np.zeros((num_rows * num_cols, num_examples))

    images_raw = array.array('B', image_file.read())
    image_file.close()

    for i in range(num_examples):
        limit1 = num_rows * num_cols * i
        limit2 = num_rows * num_cols * (i + 1)

        dataset[:, i] = images_raw[limit1: limit2]

    return dataset / 255

def loadMNISTLabels(file_name):

    label_file = open(file_name, 'rb')

    head1 = label_file.read(4)
    head2 = label_file.read(4)

    num_examples = struct.unpack('>I', head2)[0]

    labels = np.zeros((num_examples, 1), dtype=int)

    labels_raw = array.array('b', label_file.read())
    label_file.close()

    labels[:, 0] = labels_raw[:]

    return labels

def executeSoftmaxRegression():

    input_size = 784
    num_classes = 10
    lamda = 0.0001
    max_iterations = 100

    training_data = loadMNISTImages('./data/train-images.idx3-ubyte')
    training_labels = loadMNISTLabels('./data/train-labels.idx1-ubyte')

    regressor = SoftmaxRegression(input_size, num_classes, lamda)

    opt_solution = scipy.optimize.minimize(regressor.softmaxCost, np.squeeze(regressor.theta),
                                       args=(training_data, training_labels,), method='L-BFGS-B',
                                       jac=True, options={'maxiter': max_iterations})

    opt_theta = opt_solution.x

    test_data = loadMNISTImages('./data/t10k-images.idx3-ubyte')
    test_labels = loadMNISTLabels('./data/t10k-labels.idx1-ubyte')

    predictions = regressor.softmaxPredict(opt_theta, test_data)

    correct = test_labels[:, 0] == predictions[:, 0]
    print(f'Accuracy: {np.mean(correct)}')
    print(len(opt_theta))
    
    print(opt_theta.shape)

executeSoftmaxRegression()
