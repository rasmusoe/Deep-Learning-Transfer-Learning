import numpy as np

def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) where x[i] is the ith input.
  We multiply this against a weight matrix of shape (D, M) where
  D = \prod_i d_i

  Inputs:
  x - Input data, of shape (N, d_1, ..., d_k)
  w - Weights, of shape (D, M)
  b - Biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  N = x.shape[0]
  D = w.shape[0]
  X = x.reshape(N,D)
  out = X.dot(w)+b
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  # Extract dimesions and reshape input
  N = x.shape[0]
  D = w.shape[0]
  X = x.reshape(N,D)

  # Calculate derivate of the loss wrt. weights wrt. by backpropagation from upstream derivative to input
  dw = X.T.dot(dout)

  # Calculate derivative of the loss wrt. the bias
  db = np.sum(dout, axis=0, keepdims=True)  # bias nodes have no input

  # Calculate derivative of the loss wrt. input (also called 'error signal')
  dx = dout.dot(w.T).reshape(x.shape)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  # ReLU saturates all negative input values to 0 and does not affect positive 
  # input values as they are within the linear area
  out = np.maximum(x,0)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  # Backprop the ReLU non-linearity
  dx = dout
  dx[x < 0] = 0
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx

def dropout_forward(x, dropout_param):
  """
  Performs the forward pass for (inverted) dropout.

  Inputs:
  - x: Input data, of any shape
  - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter. We keep each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not in
      real networks.

  Outputs:
  - out: Array of the same shape as x.
  - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
    mask that was used to multiply the input; in test mode, mask is None.
  """
  p, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed'])

  mask = None
  out = None

  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase forward pass for inverted dropout.   #
    # Store the dropout mask in the mask variable.                            #
    ###########################################################################
    # Dropouts generates a random mask of 0's and 1's, which is used to "turn off"
    # neurons with the specified probability p. In regular dropout the activations
    # are scaled with the probability to account for the dropped networks.
    # Inverted dropout moves the scaling to the training phase instead of the 
    # testing phase.
    mask = (np.random.rand(*x.shape) < p) / p
    out = x*mask
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    ###########################################################################
    # TODO: Implement the test phase forward pass for inverted dropout.       #
    ###########################################################################
    out = x
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################

  cache = (dropout_param, mask)
  out = out.astype(x.dtype, copy=False)

  return out, cache


def dropout_backward(dout, cache):
  """
  Perform the backward pass for (inverted) dropout.

  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: (dropout_param, mask) from dropout_forward.
  """
  dropout_param, mask = cache
  mode = dropout_param['mode']
  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase backward pass for inverted dropout.  #
    # Store the dropout mask in the mask variable.                            #
    ###########################################################################
    # Dropouts generates a random mask of 0's and 1's, which is used to "turn off"
    # neurons with a specified probability p.
    dx = dout*mask
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    dx = dout
  return dx


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  # Dimensions of input data (count, channels, height, width)
  N, C, H, W = x.shape
    
  # Dimensions of filter weights array (filters, channels, height, width)
  F, C, HH, WW = w.shape
    
  # Get convolution parameters
  stride = conv_param['stride']
  padding = conv_param['pad']

  # Pad each sample (padding is only applied in the spatial dimensions H and W)
  x_pad = np.pad(x, ((0,), (0,), (padding,), (padding,)), 'constant')
    
  # Define output layer
  H_out = 1 + (H + 2 * padding - HH) / stride
  W_out = 1 + (W + 2 * padding - WW) / stride
  out = np.zeros((N, F, H_out, W_out))

  # Iterate all points in the padded input volume, multiply them by their respective weight and add bias
  for n in range(N):  # Iterate samples
      for f in range(F):  # Iterate filter kernels
          for k in range(H_out): # Iterate height dim
              for l in range(W_out): # Iterate width dim 
                  # multiply point with corresponding weight and add bias
                  out[n, f, k, l] = np.sum(x_pad[n, :, k * stride:k * stride + HH, l * stride:l * stride + WW] * w[f, :]) + b[f]
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  x, w, b, conv_param = cache
    
  # Dimensions of input data (count, channels, height, width)
  N, C, H, W = x.shape

  # Dimensions of filter weights array (filters, channels, height, width)
  F, C, HH, WW = w.shape
    
  # Dimensions of upstream derivatives (count, filters, height, width)
  N, F, H_out, W_out = dout.shape  
    
  # Get convolution parameters
  padding = conv_param['pad']
  stride = conv_param['stride']
  
  # Pad each sample (padding is only applied in the spatial dimensions H and W)  
  x_pad = np.pad(x, ((0,), (0,), (padding,), (padding,)), 'constant')

  # backpropagate gradient with respect to x  
  dx = np.zeros((N, C, H, W))
  for n in range(N):   # Iterate samples
      for i in range(H):   # Iterate input height dim
          for j in range(W):   # Iterate input width dim
              for f in range(F):   # Iterate filters
                  for k in range(H_out):   # Iterate upstream height dim
                      for l in range(W_out):   # Iterate upstream width dim
                          # Masks are created to filter the weights corresponding to padded input
                          mask1 = np.zeros_like(w[f, :, :, :])
                          mask2 = np.zeros_like(w[f, :, :, :])
                          # Filter along height dimension
                          if (i + padding - k * stride) < HH and (i + padding - k * stride) >= 0:
                              mask1[:, i + padding - k * stride, :] = 1.0
                          # Filter along width dimension
                          if (j + padding - l * stride) < WW and (j + padding - l * stride) >= 0:
                              mask2[:, :, j + padding - l * stride] = 1.0
                          # apply filtering to weight array
                          w_mask = np.sum(w[f, :, :, :] * mask1 * mask2, axis=(1, 2))
                          
                          # find derivatives by using the filtered weights 
                          dx[n, :, i, j] += dout[n, f, k, l] * w_mask    
    
  # backpropagate gradient with respect to w
  dw = np.zeros((F, C, HH, WW))
  for f in range(F):   # Iterate filters
      for c in range(C):   # Iterate channels
          for i in range(HH):    # Iterate filter height dim
              for j in range(WW):   # Iterate filter weidth dim
                  # Extract sub volume of input volumne, which corresponds to current filter
                  sub_xpad = x_pad[:, c, i:i + H_out * stride:stride, j:j + W_out * stride:stride]
                  # find derivates by multiplaction of derivates with subvolume
                  dw[f, c, i, j] = np.sum(dout[:, f, :, :] * sub_xpad)

  # backpropagate gradient with respect to b
  db = np.zeros((F))
  for f in range(F): # Iterate filters
      db[f] = np.sum(dout[:, f, :, :])
    
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  # Dimensions of input data (count, channels, height, width)
  N, C, H, W = x.shape
    
  # Get pooling parameters
  H_pool = pool_param['pool_height']
  W_pool = pool_param['pool_width']
  S_pool = pool_param['stride']
  
  # Define output layer 
  H_out = (H - H_pool) / S_pool + 1
  W_out = (W - W_pool) / S_pool + 1
  out = np.zeros((N, C, H_out, W_out))
  
  # Iterate each point in the output volume and find the maximum value in the corresponding window of the input volume
  for n in range(N):   # Iterate samples
      for c in range(C):   # Iterate channels
          for k in range(H_out):   # Iterate output height dim
              for l in range(W_out):   # Iterate output width dim
                  # find the maximum value in the corresponding window of the input volume
                  out[n, c, k, l] = np.max(x[n, c, k * S_pool:k * S_pool + H_pool, l * S_pool:l * S_pool + W_pool])
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################                      
  x, pool_param = cache
  # Dimensions of input data (count, channels, height, width)
  N, C, H, W = x.shape
    
  # Get pooling parameters
  H_pool = pool_param['pool_height']
  W_pool = pool_param['pool_width']
  S_pool = pool_param['stride']

  # Define output layer 
  H_out = (H - H_pool) / S_pool + 1
  W_out = (W - W_pool) / S_pool + 1
  dx = np.zeros((N, C, H, W))
  
  for n in range(N):
      for c in range(C):
          for k in range(H_out):
              for l in range(W_out):
                  # Extract input volume window corresponding to pooled point in output volume
                  x_pool = x[n, c, k * S_pool:k * S_pool + H_pool, l * S_pool:l * S_pool + W_pool]
                    
                  # Find maximum value in the window
                  x_pool_max = np.max(x_pool)
                  
                  # Create mask to filter all other derivatives, but the one corresponding to the maximum point
                  x_mask = x_pool == x_pool_max
                  
                  # Calculate derivative using masked derivatives
                  dx[n, c, k * S_pool:k * S_pool + H_pool, l * S_pool:l * S_pool + W_pool] += dout[n, c, k, l] * x_mask
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx

