# coding=utf-8
import numpy as np


def softmax(softmax_x):
    """Compute the softmax function for each row of the input x.

    It is crucial that this function is optimized for speed because
    it will be used frequently in later code. You might find numpy
    functions np.exp, np.sum, np.reshape, np.max, and numpy
    broadcasting useful for this task.

    Numpy broadcasting documentation:
    http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html

    You should also make sure that your code works for a single
    N-dimensional vector (treat the vector as a single row) and
    for M x N matrices. This may be useful for testing later. Also,
    make sure that the dimensions of the output match the input.

    You must implement the optimization in problem 1(a) of the
    written assignment!

    Arguments:
    x -- A N dimensional vector or M x N dimensional numpy matrix.

    Return:
    x -- You are allowed to modify x in-place
    """
    """ Softmax Regresyonu hatirlayalim: Softmax Regresyon lojistik regresyonun
     genellestirilmis halidir. Lojistik Regresyon Modeli sinif etiketi 
     y’nin olasi iki degeri için calisabilmektedir,
     Softmax Regresyon Modeli ise sinif etiketlerinin daha fazla deger 
     alabilecegi siniflandırma sorunlariyla ilgilenmektedir. """

    """ x girdisinin her satiri icin softmax fonksiyonunu hesaplayacagiz."""
    orig_shape = softmax_x.shape
    # c = constant : in practice, we make use of this property and choose c = -max_ix_i when
    # computing softmax probablities for numerical stability

    # softmax(x) = softmax(x+c)
    # softmax(x) = exp(x) / sum(exp(x))
    c = -1 * np.max(softmax_x, axis=-1, keepdims=True)
    exps = np.exp(softmax_x + c)
    sum_exps = np.sum(exps, axis=-1, keepdims=True)
    softmax_x = exps / sum_exps

    assert softmax_x.shape == orig_shape
    return softmax_x


def test_softmax_basic():
    """
    Some simple tests to get you started.
    Warning: these are not exhaustive.
    """
    print "Running basic tests..."
    test1 = softmax(np.array([1, 2]))
    print test1
    ans1 = np.array([0.26894142, 0.73105858])
    assert np.allclose(test1, ans1, rtol=1e-05, atol=1e-06)

    test2 = softmax(np.array([[1001, 1002], [3, 4]]))
    print test2
    ans2 = np.array([
        [0.26894142, 0.73105858],
        [0.26894142, 0.73105858]])
    assert np.allclose(test2, ans2, rtol=1e-05, atol=1e-06)

    test3 = softmax(np.array([[-1001, -1002]]))
    print test3
    ans3 = np.array([0.73105858, 0.26894142])
    assert np.allclose(test3, ans3, rtol=1e-05, atol=1e-06)

    print "You should be able to verify these results by hand!\n"


def test_softmax():
    """
    Use this space to test your softmax implementation by running:
        python q1_softmax.py
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    print "Running your tests..."
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE


if __name__ == "__main__":
    test_softmax_basic()
    # test_softmax()
