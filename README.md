## Assignment 1:

##### http://web.stanford.edu/class/cs224n/assignment1/index.html

In this assignment we will familiarize you with basic concepts of neural networks, word vectors, and their application to sentiment analysis.

##Setup

__Note:__ Please be sure you have Python 2.7.x installed on your system. The following instructions should work on Mac or Linux. 

__Get the code:__ Download the starter code here and the complementary written problems here.

__[Optional] virtual environment:__ Once you have unzipped the starter code, you might want to create a virtual environment for the project. If you choose not to use a virtual environment, it is up to you to make sure that all dependencies for the code are installed on your machine. To set up a virtual environment, run the following:

cd assignment1
sudo pip install virtualenv      # This may already be installed
virtualenv .env                  # Create a virtual environment
source .env/bin/activate         # Activate the virtual environment
pip install -r requirements.txt  # Install dependencies

Work on the assignment for a while ...

deactivate                       # Exit the virtual environment

__Install requirements (without a virtual environment):__ To install the required packages locally without setting up a virtual environment, run the following:

cd assignment1
pip install -r requirements.txt  # Install dependencies

__Download data:__ Once you have the starter code, you will need to download the Stanford Sentiment Treebank dataset and pretrained GloVe vectors. Run the following from the assignment1 directory:

sh get_datasets.sh

__Tasks__

There will be four parts to this assignment. 

__Q1:__ Softmax 

__Q2:__ Neural Network Basics 

__Q3:__ word2vec 

__Q4:__ Sentiment Analysis 

## Assignment 2: TensorFlow, Neural Dependency Parsing, and RNN Language Modeling

In this assignment you will learn the fundamentals of TensorFlow, use TensorFlow to implemented a feed-forward neural network for transition-based dependency parsing, and delve into backpropagation through time by computing the gradients for a recurrent neural network language model.

##Setup

__Note:__ Please be sure you have Python 2.7.x installed on your system. The following instructions should work on Mac or Linux.

__Get the code:__ Download the starter code here and the assignment handout here.

__Python package requirements:__ The core requirements for this assignment are

tensorflow
numpy

If you have a recent linux (Ubuntu 14.04 and later) install or Mac OS X, the default TensorFlow installation directions will work well for you. If not, we recommed that you work on the corn clusters. TensorFlow is already installed for the system default python on corn.

__Assignment Overview (Tasks)__

There will be three parts to this assignment.

__Q1:__ Tensorflow Softmax

__Q2:__ Neural Transition-Based Dependency Parsing 

__Q3:__ Recurrent Neural Networks: Language Modeling 

## Assignment 3: Named Entity Recognition and Recurrent Neural Networks

In this assignment you will learn about named entity recognition and implement a baseline window-based model, as well as a recurrent neural network model. The assignment also covers gated recurrent units, applying them to simple 1D sequences and the named entity recognition.

##Setup

__Note:__ Please be sure you have Python 2.7.x installed on your system. The following instructions should work on Mac or Linux.

__Get the code:__ Download the starter code here and the assignment handout here.

__Python package requirements:__ The core requirements for this assignment are

tensorflow
numpy
matplotlib

If you have a recent linux (Ubuntu 14.04 and later) install or Mac OS X, the default TensorFlow installation directions will work well for you. If not, we recommed that you work on the corn clusters. TensorFlow is already installed for the system default python on corn.

We will also be providing access to GPU computing facilities on Microsoft Azure. Details on how to access these facilities will be uploaded soon.

__Assignment Overview (Tasks)__

There will be three parts to this assignment. 

__Q1:__ A window into NER 

__Q2:__ Recurrent neural nets for NER 

__Q3:__ Grooving with GRUs

## Assignment 4: Reading Comprehension

Check out the Assignment 4 SQuAD leaderboard: http://cs224n-leaderboard.southcentralus.cloudapp.azure.com/

You are becoming a researcher in NLP with Deep Learning with this programming assignment! You will implement a neural network architecture for Reading Comprehension using the recently published Stanford Question Answering Dataset (SQuAD).

##Setup

__Note:__ Please be sure you have Python 2.7.x installed on your system. The following instructions should work on Mac or Linux.

__Get the code:__ Download the starter code here and the assignment handout here.

Python package requirements: The core requirements for this assignment are

tensorflow
nltk
tqdm
joblib

If you have a recent linux (Ubuntu 14.04 and later) install or Mac OS X, the default TensorFlow installation directions will work well for you. If not, we recommed that you work on the corn clusters. TensorFlow is already installed for the system default python on corn.

We will also be providing access to GPU computing facilities on Microsoft Azure. Details on how to access these facilities will be uploaded soon.

__Assignment Overview__

Everything you need to start doing the assignment is detailed in the assignment handout. We are not setting explicit tasks for this assignment. We only provide starter code for general APIs, data preprocessing, and evaluation.




