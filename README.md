# tensorflow
Practicing TensorFlow on MNIST data 

Feed forward neural network
input  > weight > hidden layer 1 (activation function) > weights > hidden layer 2 (activation function) > weights > output layer
compare output to intended output > cost function (cross entropy)
optimization function (optimizer) > minimize cost (AdamOptimizer, SGD, Adagrad) there are 8 options in TensorFlow
backpropagation
feed forward + backprop= epoch
As time goes on, cost will start going down after every epoch -- we will do about 10 epochs 

`TensorFlow` is a much lower level library than `scikit-learn`, use it only if necessary. 

My experience so far with both: 
1. `scikit-learn` works out of the box - set up an object of the classifier and you are good to go 
2. The main challenge with scikit-learn is the data pre-processing and setting up the data for classifier 
Example- 
`[({'I':True,'love':True,'this':True}, 'pos'),({'I':True,'dislike':True,'this':True}, 'neg')]`
3. I used `nltk` for NLP and text classification stuff - and use `SklearnClassifier` wrapper for working with text 
4. With `TensorFlow` its more low level, need to take care of commas, and lists and matrices and stuff. Lots of API changes 
I havn't looked at it yet but `Keras` sounds promising - its a high level wrapper on `TensorFlow`.  
5. I got about 95% accuracy with MNIST data, non-deep learning models should do better than this. But this was just a first attempt and the MNIST data itself is not meant for deep neural networks.  
6. The process I took with scikit-learn to do text classification for 14 categories: I did preprocessing, built learning curves, chose the number of training examples, ran the classifiers, took logistic SGD as it had confidence parameter as well, built confusion matrix, tweaked the regular expressions for preprocessing of data and used `Chartjs` along with `Flask` to do a basic web app 
I am not sure how you would do all these steps with TensorFlow - it just seems much more verbose in its function namings.. 


