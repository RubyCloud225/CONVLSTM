# ConvLSTM2D — Spatiotemporal Sequence Modelling from First Principles

A NumPy implementation of ConvLSTM2D and MaxPooling3D built without deep
learning frameworks, derived directly from the underlying recurrent
convolutional mathematics.

---

## Motivation

Standard LSTM networks process sequences as flat vectors, discarding spatial
structure entirely. ConvLSTM2D extends the LSTM formulation into two spatial
dimensions, treating each timestep as a feature map and applying convolutional
filters within the recurrent gates. This makes it naturally suited to
spatiotemporal data — weather prediction, video understanding, fluid dynamics,
and any domain where both *where* and *when* matter simultaneously.

This implementation derives the gate equations from first principles in NumPy,
with no dependency on PyTorch, TensorFlow, or any autograd framework.

---

## Architecture

### ConvLSTM2D Cell

Each cell maintains a hidden state $\mathbf{H}_t$ and cell state $\mathbf{C}_t$
across a 2D spatial grid. The four gate equations are:

$$\mathbf{I}_t = \sigma\left(\mathbf{W}_{xi} * \mathbf{X}_t + \mathbf{W}_{hi} * \mathbf{H}_{t-1} + b_i\right)$$

$$\mathbf{F}_t = \sigma\left(\mathbf{W}_{xf} * \mathbf{X}_t + \mathbf{W}_{hf} * \mathbf{H}_{t-1} + b_f\right)$$

$$\mathbf{O}_t = \sigma\left(\mathbf{W}_{xo} * \mathbf{X}_t + \mathbf{W}_{ho} * \mathbf{H}_{t-1} + b_o\right)$$

$$\mathbf{G}_t = \tanh\left(\mathbf{W}_{xg} * \mathbf{X}_t + \mathbf{W}_{hg} * \mathbf{H}_{t-1} + b_g\right)$$

where $*$ denotes 2D convolution, not elementwise multiplication. The cell and
hidden states update as:

$$\mathbf{C}_t = \mathbf{F}_t \odot \mathbf{C}_{t-1} + \mathbf{I}_t \odot \mathbf{G}_t$$

$$\mathbf{H}_t = \mathbf{O}_t \odot \tanh(\mathbf{C}_t)$$

This replaces the standard LSTM's matrix multiplications with spatial
convolutions, preserving local 2D structure through time.

### Input Tensor Shape

```
(batch, timesteps, height, width, channels)
```

Each timestep is a 2D spatial frame with `channels` feature maps. The
ConvLSTM2D layer processes the full sequence recurrently, returning the final
hidden state or the full sequence of hidden states.

### MaxPooling3D

Downsamples the 5D output tensor $(B, T, H, W, C)$ along all three spatial
and temporal dimensions using non-overlapping max pooling:

$$y_{b,t,h,w,c} = \max_{(\delta t,\, \delta h,\, \delta w)\, \in\, P} \; x_{b,\; t \cdot s_t + \delta t,\; h \cdot s_h + \delta h,\; w \cdot s_w + \delta w,\; c}$$

where $P$ is the pooling kernel and $s$ is the stride vector.

---

## Architecture Diagram

```
Input Sequence: (B, T, H, W, C)
        │
        ▼
┌───────────────────────────────────────┐
│           ConvLSTM2D Layer            │
│                                       │
│   t=0  ──► Cell ──► H_0, C_0         │
│   t=1  ──► Cell ──► H_1, C_1         │
│    ⋮            ⋮                    │
│   t=T  ──► Cell ──► H_T, C_T         │
│                                       │
│   Gates: I, F, O, G via Conv2D        │
│   Activations: σ (gates), tanh (cell) │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│            MaxPooling3D               │
│   Kernel: (2,2,2)  Stride: (2,2,2)   │
│   Output: (B, T/2, H/2, W/2, C)      │
└───────────────────────────────────────┘
        │
        ▼
  Downsampled spatiotemporal features
```

---

## Gate Equations — Expanded

To make the spatial convolution explicit, each gate weight $\mathbf{W}_{xi}$
is a 4D kernel tensor of shape
$(\text{out\_channels},\, \text{in\_channels},\, k,\, k)$
where $k$ is the kernel size. The convolution at each gate is:

$$(\mathbf{W} * \mathbf{X})_{c,h,w} = \sum_{c'} \sum_{p=0}^{k-1} \sum_{q=0}^{k-1} W_{c,c',p,q} \cdot X_{c',\, h+p,\, w+q}$$

The input gate $\mathbf{I}_t$ controls what new information enters the cell
state. The forget gate $\mathbf{F}_t$ controls what is discarded from the
previous cell state. The output gate $\mathbf{O}_t$ controls what is exposed
as the hidden state. The cell gate $\mathbf{G}_t$ produces the candidate
values to be written.

---

## Usage

```python
import numpy as np

# Instantiate ConvLSTM2D
# (in_channels, out_channels, kernel_size, padding, stride, leaky_relu_slope, num_layers)
conv_lstm = ConvLSTM2D(1, 32, 3, 1, 1, 0.2, 1)

# Input: batch=1, timesteps=10, height=32, width=32, channels=1
x = np.random.rand(1, 10, 32, 32, 1)
output = conv_lstm.forward(x)
print(output.shape)  # (1, 10, 32, 32, 32)

# Downsample with MaxPooling3D
max_pool = MaxPooling3D((2, 2, 2), (2, 2, 2))
pooled = max_pool.forward(output)
print(pooled.shape)  # (1, 5, 16, 16, 16)
```

---

## API Reference

### `ConvLSTM2D(in_channels, out_channels, kernel_size, padding, stride, leaky_relu_slope, num_layers)`

| Parameter | Type | Description |
|---|---|---|
| `in_channels` | `int` | Number of input feature maps |
| `out_channels` | `int` | Number of output feature maps (hidden state depth) |
| `kernel_size` | `int` | Spatial size of convolutional kernels in each gate |
| `padding` | `int` | Zero-padding applied to spatial dimensions |
| `stride` | `int` | Convolutional stride |
| `leaky_relu_slope` | `float` | Negative slope for LeakyReLU activation |
| `num_layers` | `int` | Number of stacked ConvLSTM2DCell layers |

**Methods**

- `forward(x)` — runs the full input sequence through all stacked cells,
  returning hidden states of shape `(B, T, H, W, out_channels)`

### `MaxPooling3D(pool_size, stride)`

| Parameter | Type | Description |
|---|---|---|
| `pool_size` | `tuple(int, int, int)` | Pooling kernel dimensions `(t, h, w)` |
| `stride` | `tuple(int, int, int)` | Stride in each dimension `(t, h, w)` |

**Methods**

- `forward(x)` — applies 3D max pooling to a 5D input tensor
  `(B, T, H, W, C)`, returning a downsampled tensor

---

## Why No Framework?

Building this in NumPy forces explicit derivation of every operation —
convolution, gate activations, state updates, pooling. There is no autograd
graph, no implicit broadcasting magic, no framework abstraction hiding what
the equations actually do. The result is a direct correspondence between the
mathematics above and the code in `model/`.

---

## Dependencies

```
numpy
```

No PyTorch. No TensorFlow. No deep learning framework of any kind.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

*Original implementation by Catherine Earl, 2025.*
