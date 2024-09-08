import numpy as np

class MaxPooling3D:
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride
    
    def forward(self, x):
        batch_size, channels, depth, height, width = x.shape
        pooled_depth = (depth - self.pool_size[0] // self.stride[0] + 1)
        pooled_height = (height - self.pool_size[1] // self.stride[1] + 1)
        pooled_width = (width - self.poolsize[2] // self.stride[2] + 1)

        output 