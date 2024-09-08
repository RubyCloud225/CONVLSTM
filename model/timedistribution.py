import numpy as np

class TimeDistributionCalculator:
    def __init__(self, dropout_rate, num_timesteps):
        """
        Initialize the TimeDistributionCalculator

        args:
            dropout_rate (float): The dropout rate
            num_timesteps (int): The number of timesteps
        """
        self.dropout_rate = dropout_rate
        self.num_timesteps = num_timesteps
    
    def calculate_time_distribution(self):
        """
        Calculates the time distribution for the dropout

        Returns: 
            np.ndarray: The time distribution.

        """
        time_distribution = np.random.binomial(1, 1 - self.dropout_rate, self.num_timesteps)
        return time_distribution
    
    def apply_dropout(self, x):
        """
        Applies the dropout at the input tensor.

        args:
            x (np.ndarray): The input tensor

        returns:
            np.ndarray: The tensor after applying dropout.
        """

        time_distribution = self.calculate_time_distribution()
        x_dropped = x * time_distribution[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
        return x_dropped
    
"""
import numpy as np

# Create a TimeDistributionCalculator instance
calculator = TimeDistributionCalculator(0.2, 10)

# Create a random input tensor
x = np.random.rand(1, 10, 32, 32, 32)

# Apply the dropout
x_dropped = calculator.apply_dropout(x)
print(x_dropped.shape)  # Output: (1, 10, 32, 32, 32)
"""