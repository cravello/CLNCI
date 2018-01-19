import numpy as np

from dataset import mnist


### Function definitions.

def convolution(input_feature_map, kernel, zero_padding="no", stride=1):
    """A two dimensional convolution over an square image.
    
    Args:
        input_feature_map (2D numpy.ndarray): The input image, must be square.
        kernel (2D numpy.ndarray)           : The kernel used on the convolution, must be square.
        stride (int > 0)                    : The stride step, same on both axes.
        zero_padding (string || int >= 0)   : The zero padding type or length,
                                              same on each side and on both axes.
                                              This can be:
                                                  'no',
                                                   False,
                                                   None   : No zero padding, same as zero_padding=0.
                                                  'half'  : Half zero padding. The output feature map will be 
                                                            of the same dimension as the input feature map.
                                                            The kernel dimension must be odd.
                                                  'full'  : Full zero padding.
                                                   n (int): An arbitrary zero padding of length 'n'.

    Returns:
        Return a two dimensional numpy.ndarray convolved image,
        the output feature map of the kernel used.
    """
    #TODO
    # Check for input square.
    # Check for kernel square.
    # Check for stride > 0
    
    i = input_feature_map.shape[0]
    k = kernel.shape[0]
    s = stride
    
    if zero_padding == "no" or zero_padding == False or zero_padding is None:
        p = 0
    elif zero_padding == "half":
        #TODO If kernel dimension is not odd raise exception.
        p = int((k - 1)/2)
    elif zero_padding == "full":
        p = int(k - 1)
    elif type(zero_padding) == int:
        p = zero_padding
    else:
        pass #TODO Raise exception.
    
    o = int(np.floor((i + 2*p - k)/s) + 1)
    output_shape = (o,o)
    
    padded_input_feature_map = np.pad(input_feature_map, p, "constant")
    
    output_feature_map = np.zeros(output_shape)

    for x in range(o):
        for y in range(o):
            output_feature_map[x][y] = np.sum(padded_input_feature_map[x*s:x*s+k, y*s:y*s+k]*kernel)

    return output_feature_map


def pooling(input_feature_map, dimension, stride=1, function=np.max):
    """Pooling over a convolved image
    
    Args:
        input_feature_map (2D numpy.ndarray): The pooling will be applied over this input feature map,
                                              usually this is the output of a convolution step.
        dimension (int>0)                   : The dimension of the square window for pooling.
        stride (int>0)                      : The stride step, same on both axes.
        function (numpy.ndarray->float)     : The pooling function, the default is numpy.max.

    Returns:
        Returns a 2D numpy.ndarray as the pooled output feature map.
    """
    i = input_feature_map.shape[0]
    d = dimension
    s = stride
    
    o = int(np.floor((i - d)/s) + 1)
    output_shape = (o,o)
    
    output = np.zeros(output_shape)
    
    for x in range(o):
        for y in range(o):
            output[x][y] = function(input_feature_map[x*s:x*s+d, y*s:y*s+d])

    return output


def relu(image):
    """Simple implementation of a ReLU.
    """
    return np.maximum(image, 0, image)


def sigmoid(array, function_name="logistic"):
    """Implementation of some sigmoid functions.
    
    Args:
        array (numpy.ndarray)          : numpy.ndarray input of 'function'.
        function (numpu.ndarray->float): The name of the sigmoid function to use.
                                         One of the following: 'logistic',
                                                               'tanh'
    Returns:
        The function 'function' applied to 'array'.
    """
    if function_name == "logistic":
        return 1/(1 + np.exp(-ndarray))
    elif function_name == "tanh":
        return np.tanh(array)
    else:
        pass #TODO: Raise exception.



### Classes definition.

class ConvolutionalLayer:
    """Implementation of a Convolutional Layer for a two dimensional input feature maps.
    
    Note: See __init__(...) for initialization docs.
    
    A Convolutional Layer consists of a convolutional step, a pooling step, and usually a rectification step.

    Class atributes:
        self.input_feature_maps  : The feature maps (a.k.a. images) passed as input to the layer.
        self.output_feature_maps : The feature maps, result of the layer. The output of the conv.
        self.kernels             : The kernels for the convolutional step.
        self.dimension           : The dimension of the (square) images.
        self.kernel_dimension    : The dimension of the (square) kernels.
    
    Class methods:
        self.convolution(...)
        self.pooling(...)
        self.relu(...)

    Example:
        cl1 = ConvolutionalLayer(images, kernels1) # Images and kernels iterables as inputs.
        cl1.convolution("half", 2)                 # Convolution step with half padding and a stride of 2.
        cl1.pooling(2)                             # Pooling step.
        cl1.relu()                                 # Application of ReLU.
        
        cl2 = ConvolutionalLayer(cl1(), kernels2)  # cl1() as the input feature maps of a new convolutional layer,
                                                   # and new kernels.
        cl2.convolution("no", 2)                   # A convolutional step with no zero padding, and a stride of 2.
        cl2.pooling(3, np.mean)                    # The pooling step, with a window of size 3 and an average
                                                   # pooling function.
        
        flc = FullyConnectedLayer(cl2.(),...) 
        ... etc
    """
    def __init__(self, input_feature_maps, kernels):
        """
        Args:
            input_feature_maps (numpy.ndarray, iterable):
                An iterable of 2D numpy.ndarray representing the input images,
                for instance a 3D numpy.ndarraym or a list of 2D numpy.ndarrays.
            
            kernels (numpy.ndarrays, iterable) :
                An iterable of 2D numpy.ndarrays,
                for instance a 3D numpy.ndarray, or a list of 2D numpy.ndarrays.

        Example:
            TODO
        """
        self.input_feature_maps = np.stack(input_feature_maps)
        self.output_feature_maps = None
        self.kernels = np.stack(kernels)
        
        self.dimension = self.input_feature_maps.shape[0]
        self.kernels_dimension = self.kernels.shape[0]


    def __call__(self):
        return self.output_feature_maps


    def convolution(self, zero_padding="no", stride=1):
        """
        Args:
            TODO
        """
        for i in range(self.dimension): # i index over input images of the CNN.
            for kernel in self.kernels:
                convolution_result = convolution(self.input_feature_maps[i], kernel, zero_padding, stride)
                convolution_result = convolution_result[np.newaxis,:,:]
                if self.output_feature_maps is None:
                    self.output_feature_maps = convolution_result
                else:
                    self.output_feature_maps = np.concatenate((self.output_feature_maps, convolution_result), axis=0)

        return self


    def pooling(self, dimension, stride=1, function=np.max):
        """
        Args:
            TODO
        """
        temp_feature_maps = None
        for i in range(self.output_feature_maps.shape[0]):
            pooling_result = pooling(self.output_feature_maps[i], dimension, stride, function)
            pooling_result = pooling_result[np.newaxis,:,:]
            if temp_feature_maps is None:
                temp_feature_maps = pooling_result
            else:
                temp_feature_maps = np.concatenate((temp_feature_maps, pooling_result), axis=0)

        self.output_feature_maps = temp_feature_maps
        
        return self


    def relu(self):
        """
        Args:
            TODO
        """
        for i in range(self.output_feature_maps.shape[0]):
            self.output_feature_maps[i] = relu(self.output_feature_maps[i])

        return self


class FullyConnectedLayer:
    """
    TODO TODO TODO TODO TODO
    """
    def __init__(self):
        pass


if __name__ == "__main__":
    # Code example.
    path = "/data/rsrc/data/mnist/"

    images = mnist.load_train_images(path)
    
    kernels = []
    kernels.append(np.identity(7))
    kernels.append(np.rot90(np.identity(7)))
    kernels.append(np.array([[0,0,0,1,0,0,0]]).repeat(7, axis=0))
    kernels.append(np.rot90(np.array([[0,0,0,1,0,0,0]]).repeat(7, axis=0)))
    
    cl1 = ConvolutionalLayer(images[:4], kernels)
    cl1.convolution("half")
    cl1.pooling(3, 2)
    cl1.relu()
    
    cl2 = ConvolutionalLayer(cl1(), kernels)
    cl2.convolution("half")
    cl2.pooling(3, 2, np.mean)

    cl2()
