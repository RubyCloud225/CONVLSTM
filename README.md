ConvLSTM2D and MaxPooling3D Implementation
This repository contains a NumPy implementation of a ConvLSTM2D layer and a MaxPooling3D layer.

ConvLSTM2D
The ConvLSTM2D layer is a convolutional LSTM layer that can be used for spatiotemporal data. It consists of multiple ConvLSTM2DCell instances stacked together.

MaxPooling3D
The MaxPooling3D layer is a 3D max pooling layer that can be used to downsample 5D input tensors.

Installation
To use this implementation, simply clone this repository and use the ConvLSTM2D and MaxPooling3D classes in your Python code.

Example Usage
python

Verify

Open In Editor
Edit
Copy code
import numpy as np

# Create a ConvLSTM2D instance
conv_lstm = ConvLSTM2D(1, 32, 3, 1, 1, 0.2, 1)

# Create a random input tensor
x = np.random.rand(1, 10, 32, 32, 1)

# Apply the ConvLSTM2D layer
output = conv_lstm.forward(x)
print(output.shape)  # Output: (1, 10, 32, 32, 32)

# Create a MaxPooling3D instance
max_pool = MaxPooling3D((2, 2, 2), (2, 2, 2))

# Apply the MaxPooling3D layer
pooled_output = max_pool.forward(output)
print(pooled_output.shape)  # Output: (1, 10, 16, 16, 16)
Documentation
ConvLSTM2D
__init__(in_channels, out_channels, kernel_size, padding, stride, leaky_relu_slope, num_layers): Initializes the ConvLSTM2D layer.
in_channels: The number of input channels.
out_channels: The number of output channels.
kernel_size: The size of the convolutional kernel.
padding: The amount of padding to apply.
stride: The stride of the convolutional kernel.
leaky_relu_slope: The slope of the leaky ReLU activation function.
num_layers: The number of ConvLSTM2DCell instances to stack.
forward(x): Applies the ConvLSTM2D layer to the input tensor x.
x: The input tensor.
MaxPooling3D
__init__(pool_size, stride): Initializes the MaxPooling3D layer.
pool_size: The size of the pooling kernel.
stride: The stride of the pooling kernel.
forward(x): Applies the MaxPooling3D layer to the input tensor x.
x: The input tensor.
License
This implementation is licensed under the MIT License. See LICENSE for details.
