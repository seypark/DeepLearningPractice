# This is my implementation for mnist classification with LR.
# I hope it generates similar result from original implementation from tutorial
# # http://deeplearning.net/tutorial/logreg.html (sgd_optimization_mnist.py)
# I borrow load_data function from original code


import theano
import os
import theano.tensor as T
import gzip
import cPickle
import numpy

class LR(object):
    def __init__(self, input, n_in, n_out):

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

    def loss(self, y):

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
    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """
        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

def load_data(dataset):
    # check dataset file exists

    if os.path.isfile(dataset):
        f = gzip.open(dataset, 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()

        def shared_dataset(data_xy, borrow=True):
           """ Function that loads the dataset into shared variables
           The reason we store our dataset in shared variables is to allow
           Theano to copy it into the GPU memory (when code is run on GPU).
           Since copying data into the GPU is slow, copying a minibatch everytime
           is needed (the default behaviour if the data is not in a shared
           variable) would lead to a large decrease in performance.
           """
           data_x, data_y = data_xy
           shared_x = theano.shared(numpy.asarray(data_x,
                                                  dtype=theano.config.floatX),
                                    borrow=borrow)
           shared_y = theano.shared(numpy.asarray(data_y,
                                                  dtype=theano.config.floatX),
                                    borrow=borrow)
           # When storing data on the GPU it has to be stored as floats
           # therefore we will store the labels as ``floatX`` as well
           # (``shared_y`` does exactly that). But during our computations
           # we need them as ints (we use labels as index, and if they are
           # floats it doesn't make sense) therefore instead of returning
           # ``shared_y`` we will have to cast it to int. This little hack
           # lets ous get around this issue
           return shared_x, T.cast(shared_y, 'int32')

        test_set_x, test_set_y = shared_dataset(test_set)
        valid_set_x, valid_set_y = shared_dataset(valid_set)
        train_set_x, train_set_y = shared_dataset(train_set)

    else:
        print "cannot find dataset file %s" % dataset
        return false


    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    return rval

def mnist_classification(dataset='mnist.pkl.gz', batch_size = 600, learning_rate=0.13):
    #load data
    datasets = load_data(dataset)
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')
    y = T.ivector('y')
    classifier = LR(input=x, n_in=28 * 28, n_out=10)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    #split training set
    n_train_set = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_set = test_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_set = valid_set_x.get_value(borrow=True).shape[0] / batch_size

    n_test = test_set_x.get_value(borrow=True).shape[0]
    n_valid = valid_set_x.get_value(borrow=True).shape[0]
    #loss for logistic regression is usually neg-log likelihood
    cost = classifier.loss(y)
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    train_function = theano.function(
        inputs=[index],
        outputs=cost,
        updates=[(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)],
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    test_function = theano.function(
        inputs=[],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[1:n_test-1],
            y: test_set_y[1:n_test-1]
        }
    )

    valid_function = theano.function(
        inputs=[],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[1:n_valid-1],
            y: valid_set_y[1:n_valid-1]
        }
    )

    best_valid_loss = valid_set_x.get_value(borrow=True).shape[1]
    stop_train = False
    min_iter = 200
    iter = 0
    while (not stop_train):
        iter = iter +1
        #train with training set
        for i in xrange(n_train_set):
            # do gradient descent with logistic regression
            # using each batch
            
            train_cost = train_function(i)

            # after train n_train_set, do validation
            if i==n_train_set-1:
                # evaluate validation set
                
                valid_loss = valid_function()

                print 'validation error %s %%' % (valid_loss)
                if valid_loss < best_valid_loss :
                    best_valid_loss = valid_loss
                    stop_train = False
                else:
                    if(iter > min_iter):
                        stop_train = True

    test_loss = test_function()
    print 'test error %s %%' % (test_loss)


if __name__ == '__main__':
    mnist_classification();