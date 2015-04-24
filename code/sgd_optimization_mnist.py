
import cPickle
import theano
import theano.tensor as T
import numpy
import gzip

import os

class LogisticRegression(object):
    def __init__():

        #Define weight vector n_in and n_out is dimension of input and output vector
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )

        #Define vector for bias term b. length is n_out
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]

    # define a (symbolic) cost variable to minimize, using the instance method classifier.negative_log_likelihood.
    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    # cost = classifier.negative_log_likelihood(y)

    # Learning optimal model parameters involves minimizing a loss function. 
    # In the case of multi-class logistic regression, it is very common to use the negative log-likelihood as the loss.
    # This is equivalent to maximizing the likelihood of the data set \cal{D} under the model parameterized by \theta. 
    # Let us first start by defining the likelihood \cal{L} and loss \ell:

    def negative_log_likelihood(self, y):

        # y.shape[0] is (symbolically) the number of rows in y, i.e., # of example
        # T.arange(n) is a symbolic vector which will contain [0,1,2,... n-1]
        # 
        # T.log(self.p_y_given_x) is a matrix of Log-Probabilities (LP) with 
        # one row per example and one column per class 
        #
        # LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ..., LP[n-1,y[n-1]]]
        # T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

def load_data(dataset):
    # check dataset file exists

    if os.path.isfile(dataset):
        f = gzip.open(dataset, 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()

        test_set_x, test_set_y = test_set
        valid_set_x, valid_set_y = valid_set
        train_set_x, train_set_y = train_set
    else:
    	print "cannot find dataset file %s" % dataset
    	train_set_x = []
    	train_set_y = []
    	test_set_x = []
    	test_set_y = []
    	valid_set_x = []
    	valid_set_y = []


    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    return rval

def train(dataset='mnist.pkl.gz', batch_size=600):
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.shape[0] / batch_size
    n_valid_batches = valid_set_x.shape[0] / batch_size
    n_test_batches = test_set_x.shape[0] / batch_size
    
if __name__ == '__main__':
	train()