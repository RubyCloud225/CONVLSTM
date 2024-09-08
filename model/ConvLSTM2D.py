import numpy as np
from timedistribution import TimeDistributionCalculator

class ConvLSTM2DCell:
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, leaky_relu_slope):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.leaky_relu_slope = leaky_relu_slope
        self.weights = self._init_weights()


    def _init_weights(self):
        weights = {}
        weights['W_xi'] = np.random.rand(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        weights['W_hi'] = np.random.rand(self.out_channels, self.out_channels, self.kernel_size, self.kernel_size)
        weights['W_xf'] = np.random.rand(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        weights['W_hf'] = np.random.rand(self.out_channels, self.out_channels, self.kernel_size, self.kernel_size)
        weights['W_xc'] = np.random.rand(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        weights['W_hc'] = np.random.rand(self.out_channels, self.out_channels, self.kernel_size, self.kernel_size)
        weights['W_xo'] = np.random.rand(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        weights['W_ho'] = np.random.rand(self.out_channels, self.out_channels, self.kernel_size, self.kernel_size)
        weights['b_i'] = np.zeros((self.out_channels, 1, 1))
        weights['b_f'] = np.zeros((self.out_channels, 1, 1))
        weights['b_c'] = np.zeros((self.out_channels, 1, 1))
        weights['b_o'] = np.zeros((self.out_channels, 1, 1))
        return weights

    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _tanh(self, x):
        return np.tanh(x)

    def _leaky_relu(self, x):
        return np.maximum(x, self.leaky_relu_slope * x)

    def forward(self, x, h, c):
        i = self._sigmoid(np.convolve(x, self.weights['W_xi'], mode='valid') + np.convolve(h, self.weights['W_hi'], mode='valid') + self.weights['b_i'])
        f = self._sigmoid(np.convolve(x, self.weights['W_xf'], mode='valid') + np.convolve(h, self.weights['W_hf'], mode='valid') + self.weights['b_f'])
        g = self._tanh(np.convolve(x, self.weights['W_xc'], mode='valid') + np.convolve(h, self.weights['W_hc'], mode='valid') + self.weights['b_c'])
        o = self._sigmoid(np.convolve(x, self.weights['W_xo'], mode='valid') + np.convolve(h, self.weights['W_ho'], mode='valid') + self.weights['b_o'])
        c_new = f * c + i * g
        h_new = o * self._tanh(c_new)
        return h_new, c_new

class ConvLSTM2D:
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, leaky_relu_slope, num_layers):
        self.cells = [ConvLSTM2DCell(in_channels, out_channels, kernel_size, padding, stride, leaky_relu_slope) for _ in range(num_layers)]

    def forward(self, x):
        h = np.zeros((x.shape[0], x.shape[2], x.shape[3], x.shape[1]))
        c = np.zeros((x.shape[0], x.shape[2], x.shape[3], x.shape[1]))
        outputs = []
        for i in range(x.shape[1]):
            h, c = self.cells[0].forward(x[:, i, :, :, :], h, c)
            outputs.append(h)
        outputs = np.stack(outputs, axis=1)
        outputs = TimeDistributionCalculator.apply_dropout(outputs)
        return outputs 
