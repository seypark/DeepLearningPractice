# This code is from the web tutorial
# http://deeplearning.net/tutorial/logreg.html
import cPickle
import theano
import theano.tensor as T
import numpy
import gzip
import time
import sys
import os

class LogisticRegression(object):
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
    def errors(self, y):
        print 'error call'
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """
        print self.y_pred.ndim
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

def sgd_optimization_mnist(dataset='mnist.pkl.gz', batch_size=600, learning_rate=0.13, n_epochs=1000):
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size
    
    #if train_set is 60000, batch_size is 6000. n_train_batches is 10;

    ##                ##   
    ## Building Model ##
    ##                ##
    index = T.lscalar()  # index to a [mini]batch

    x = T.matrix('x')
    y = T.ivector('y')


    # n_in is image size 28 by 28
    # n_out is the number of class (0,..., 9)
    classifier = LogisticRegression(input=x, n_in=28 * 28, n_out=10)
    cost = classifier.negative_log_likelihood(y)

    # compiling a Theano function that computes the mistakes that are made by
    # the model on a minibatch
    
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
   
    # compute the gradient of cost with respect to theta = (W,b)
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    # start-snippet-3
    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]

    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-3

    ###############
    # TRAIN MODEL #
    ###############
    print '... training the model'
    # early-stopping parameters
    patience = 5000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                                  # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                  # considered significant

    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch


    print 'n_train_batches %d' % n_train_batches
    print 'validation_frequency %d' % validation_frequency
    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            #iter is 0,... ,83 for epoch = 1
            #iter is 84,... ,166 for epoch = 2


            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i)
                                     for i in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    # test it on the test set

                    test_losses = [test_model(i)
                                   for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(
                        (
                            '     epoch %i, minibatch %i/%i, test error of'
                            ' best model %f %%'
                        ) %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            test_score * 100.
                        )
                    )

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print(
        (
            'Optimization complete with best validation score of %f %%,'
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., test_score * 100.)
    )
    print 'The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.1fs' % ((end_time - start_time)))

if __name__ == '__main__':
	sgd_optimization_mnist()