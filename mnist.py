"""Builds the MNIST network.
Implements the inference/loss/training pattern for model building.
1. inference() - Builds the model as far as is required for running the network
forward to make predictions.
2. loss() - Adds to the inference model the layers required to generate loss.
3. training() - Adds to the loss model the Ops required to generate and
apply gradients.
This file is used by the various "fully_connected_*.py" files and not meant to
be run.
This file is used by the various "fully_connected_*.py" files and not meant to
be run.
"""
import numpy as np
import tensorflow as tf

# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

def xavier_weight_init():
  """
  Returns function that creates random tensor.

  The specified function will take in a shape (tuple or 1-d array) and must
  return a random tensor of the specified shape and must be drawn from the
  Xavier initialization distribution.

  Hint: You might find tf.random_uniform useful.
  """
  def _xavier_initializer(shape, **kwargs):
    """Defines an initializer for the Xavier distribution.

    This function will be used as a variable scope initializer.

    https://www.tensorflow.org/versions/r0.7/how_tos/variable_scope/index.html#initializers-in-variable-scope

    Args:
      shape: Tuple or 1-d array that species dimensions of requested tensor.
    Returns:
      out: tf.Tensor of specified shape sampled from Xavier distribution.
    """
    ### YOUR CODE HERE
    epsilon = np.sqrt(6.0) / np.sqrt(sum(shape))
    out = tf.random_uniform(shape=shape, minval = -epsilon, maxval = epsilon)
    ### END YOUR CODE
    return out
  # Returns defined initializer function.
  return _xavier_initializer

def inference(images, hidden1_units, hidden2_units):
    """Build the MNIST model up to where it may be used for inference.
      Args:
        images: Images placeholder, from inputs().
        hidden1_units: Size of the first hidden layer.
        hidden2_units: Size of the second hidden layer.
      Returns:
        softmax_linear: Output tensor with the computed logits.
    """
    with tf.variable_scope('hidden1', initializer=xavier_weight_init()):
        weights = tf.get_variable('weights', [IMAGE_PIXELS, hidden1_units])
        biases = tf.get_variable('biases', [hidden1_units])
        hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)

    with tf.variable_scope('hidden2', initializer=xavier_weight_init()):
        weights = tf.get_variable('weights', [hidden1_units, hidden2_units])
        biases = tf.get_variable('biases', [hidden2_units])
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

    with tf.variable_scope('softmax_linear', initializer=xavier_weight_init()):
        weights = tf.get_variable('weights', [hidden2_units, NUM_CLASSES])
        biases = tf.get_variable('biases', [NUM_CLASSES])
        logits = tf.matmul(hidden2, weights) + biases

    return logits

def loss(logits, labels):
    """ Calculates the loss from the logits and the labels
    Args:
        logits: Logits tensor, float - [batch_size, NUM_CLASSES]
        labels: Labels tensor, int32 - [batch_size]
    Returns:
        loss: Loss tensor of type float
    """
    labels = tf.to_int64(labels)
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits, labels), name='xentropy_mean')
    return cross_entropy

def training(loss, learning_rate):
    """Sets up the training Ops.
    Creates a summarizer to track the loss over time in TensorBoard.
    Creates an optimizer and applies the gradients to all trainable variables.
    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train.
    Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.
    Returns:
    train_op: The Op for training.
    """
    # Add a scalar summary for the snapshot loss.
    tf.scalar_summary(loss.op.name, loss)
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

def evaluation(logits, labels):
    """Evaluate the quality of the logits at predicting the label.
    Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size], with values in the
      range [0, NUM_CLASSES).
    Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
    """
    # For a classifier model, we can use the in_top_k Op.
    # It returns a bool tensor with shape [batch_size] that is true for
    # the examples where the label is in the top k (here k=1)
    # of all logits for that example.
    correct = tf.nn.in_top_k(logits, labels, 1)
    # Return the number of true entries.
    return tf.reduce_sum(tf.cast(correct, tf.int32))
