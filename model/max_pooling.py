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

        output = np.zeros((batch_size, channels, pooled_depth, pooled_height, pooled_width))

        for b in range(batch_size):
            for c in range(channels):
                for d in range(pooled_depth):
                    for h in range(pooled_height):
                        for w in range(pooled_width):
                            d_start = d * self.stride[0]
                            d_end = min(d_start + self.pool_size[0], depth)
                            h_start = h * self.stride[1]
                            h_end = min(h_start + self.pool_size[1], height)
                            w_start = w * self.stride[2]
                            w_end = min(w_start + self.pool_size[2], width)

                            output[b, c, d, h, w] = np.max(x[b, c, d_start:d_end, h_start:h_end, w_start:w_end])
        return output
"""
#Example Useage
max_pool = MaxPooling3D((2, 2, 2), (2, 2, 2))
x = np.random.rand(1, 10, 32, 32, 32)
pooled_x = max_pool.forward(x)
print(pooled_x.shape)  # Output: (1, 10, 16, 16, 16)  
"""