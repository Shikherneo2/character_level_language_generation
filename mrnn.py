import tensorflow as tf
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import RNNCell

def ntanh(_x, name="normalizing_tanh"):
    # From comments on reddit, the normalizing tanh function:
    # 1.5925374197228312
    scale = 1.5925374197228312
    return scale*tf.nn.tanh(_x, name=name)


class MRNNCell(RNNCell):
  def __init__(self, num_units, depth=2, forget_bias=-2.0, activation=ntanh):

    assert activation.__name__ == "ntanh"
    self._num_units = num_units
    self._in_size = num_units
    self._depth = depth
    self._forget_bias = forget_bias
    self._activation = activation
    
  @property
  def input_size(self):
    return self._in_size

  @property
  def output_size(self):
    return self._num_units

  @property
  def state_size(self):
    return self._num_units

  def __call__(self, inputs, state, timestep=0, scope=None):
    current_state = state

    for i in range(self._depth):
      with tf.variable_scope('h_'+str(i)):
        if i == 0:
          h = self._activation(
            multiplicative_integration([inputs,current_state], self._num_units))
        else:
          h = tf.layers.dense(current_state, self._num_units, self._activation, 
                bias_initializer=tf.zeros_initializer())

      with tf.variable_scope('gate_'+str(i)):
        if i == 0:
          t = tf.sigmoid(
            multiplicative_integration([inputs,current_state], self._num_units, 
              self._forget_bias))

        else:
          t = tf.layers.dense(current_state, self._num_units, tf.sigmoid, 
                bias_initializer=tf.constant_initializer(self._forget_bias))

      current_state = (h - current_state)* t + current_state

    return current_state, current_state

def multiplicative_integration(list_of_inputs, output_size, initial_bias_value=0.0,
  weights_already_calculated=False, scope=None):
    """Multiplicative Integration from https://arxiv.org/abs/1606.06630
    expects len(2) for list of inputs and will perform integrative multiplication
    weights_already_calculated will treat the list of inputs as Wx and Uz and is useful for batch normed inputs
    """
    with tf.variable_scope(scope or 'double_inputs_multiple_integration'):
      if len(list_of_inputs) != 2: 
        raise ValueError('list of inputs must be 2, you have: {}'.format(len(list_of_inputs)))
      
      if weights_already_calculated:
        Wx = list_of_inputs[0]
        Uz = list_of_inputs[1]

      else:
        with tf.variable_scope('Calculate_Wx_mulint'):
          Wx = tf.layers.dense(list_of_inputs[0], output_size, use_bias=False)

        with tf.variable_scope("Calculate_Uz_mulint"):
          Uz = tf.layers.dense(list_of_inputs[1], output_size, use_bias=False)

      with tf.variable_scope("multiplicative_integration"):
        alpha = tf.get_variable('mulint_alpha', [output_size], 
            initializer = tf.truncated_normal_initializer(mean=1.0, stddev=0.1))

        # For efficiency, we retrieve both beta parameters via tf split
        beta1, beta2 = tf.split(
          tf.get_variable('mulint_params_betas', [output_size*2], 
            initializer = tf.truncated_normal_initializer(mean=0.5, stddev=0.1)),
          num_or_size_splits=2,
          axis=0) 

        original_bias = tf.get_variable('mulint_original_bias', [output_size], 
            initializer = tf.truncated_normal_initializer(mean=initial_bias_value, stddev=0.1)) 

      final_output = alpha*Wx*Uz + beta1*Uz + beta2*Wx + original_bias

    return final_output