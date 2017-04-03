#We will use MNIST data for digits -- handwritten digits prediction

#Feed forward neural network
# input  > weight > hidden layer 1 (activation function) > weights > hidden layer 2 (activation function) > weights > output layer
# compare output to intended output > cost function (cross entropy)
# optimization function (optimizer) > minimize cost (AdamOptimizer, SGD, Adagrad) there are 8 options in TensorFlow
# backpropagation
# feed forward + backprop= epoch
# As time goes on, cost will start going down after every epoch -- we will do about 10 epochs

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)   #One is on and rest is off

# We have 10 classes with 0 to 9 - so 0 is actually [1,0,0,0,0,0,0,0,0,0,0], 1 will be [0,1,0,0,0,0,0,0,0,0] and so on - this is one hot

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100  #Feed 100 images at a time

# height x width
x = tf.placeholder('float', [None,784])   #784 values from 28x28 pixels  to flatten
y = tf.placeholder('float')   #Label of the data

def neural_network_model(data):
    #Create a random weights for each pixel for first hidden layer
    # (input data * weights) + biases
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))} #Even if all inputs are zero, some neurons can fire
    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}
    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}
    output_layer =   {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                      'biases':tf.Variable(tf.random_normal([n_classes]))}

    #First layer
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)    #rectified linear for threshold function

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']),hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y)) #Calculate cost using cross entropy

    #learning rate can be modified in Adam Optimizer (like SGD or AdaGrad) -- it is set at 0.0001
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    #Cycles feed forward + backprop
    hm_epochs = 10

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        #Train the data
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x , epoch_y = mnist.train.next_batch(batch_size)   #Chunk the data into batch size
                _, c = session.run([optimizer,cost], feed_dict={x : epoch_x, y : epoch_y})
                epoch_loss += c
            print 'Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss

        #Compare prediction to actual label
        correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))

        print 'Accuracy:', accuracy.eval({x:mnist.test.images,y:mnist.test.labels})


train_neural_network(x)



# x1 = tf.constant(5)
# x2 = tf.constant(3)

# result = tf.multiply(x1,x2)
# print result

# session = tf.Session()

# with tf.Session() as session:
#     print session.run(result)