# Guide: 04 Building a Multi-Layer Network

This guide demonstrates how to define a simple multi-layer neural network by stacking layers within an `nn.Module`, as shown in `04_multi_layer_network.py`.

**Core Concept:** Most practical neural networks consist of multiple layers stacked together. By combining linear transformations (`nn.Linear`) with non-linear activation functions (`nn.ReLU`, etc.), networks can learn hierarchical representations and model complex relationships in data.

## Typical Multi-Layer Structure (MLP)

A common type of feed-forward network is the Multi-Layer Perceptron (MLP). It typically consists of:

1. **Input Layer:** Often implicitly defined by the input dimension passed to the first `nn.Linear` layer.
2. **Hidden Layer(s):** One or more layers that perform intermediate computations. Each hidden layer usually consists of:
    - A linear transformation (`nn.Linear`).
    - A non-linear activation function (e.g., `nn.ReLU`).
3. **Output Layer:** The final layer (usually `nn.Linear`) that produces the network's output in the desired format (e.g., class scores, regression values).

## Connecting Layers

A critical aspect is ensuring the dimensions match between consecutive layers:

- The `in_features` of a linear layer must match the number of features coming _out_ of the previous step in the `forward` pass.
- The `out_features` of a linear layer determines the number of features passed _into_ the next step.
- Activation functions typically don't change the shape/number of features.

## Example: `MultiLayerNet`

The script defines a network with one input layer, one hidden layer (Linear + ReLU), and one output layer:

```python
# Script Snippet (Class Definition):
import torch
import torch.nn as nn

class MultiLayerNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MultiLayerNet, self).__init__()

        # Define the layers sequentially
        # Layer 1: Input features -> Hidden features
        self.layer_1 = nn.Linear(input_size, hidden_size)
        # Activation for Layer 1 output
        self.relu = nn.ReLU()
        # Layer 2: Hidden features -> Output features
        self.layer_2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Define the data flow through the layers
        # 1. Pass through first linear layer
        out = self.layer_1(x)
        # 2. Apply activation function
        out = self.relu(out)
        # 3. Pass through second linear layer
        out = self.layer_2(out)
        return out
```

- **`__init__` Breakdown:**
  - `self.layer_1`: Takes `input_size` features and outputs `hidden_size` features.
  - `self.relu`: The ReLU activation instance.
  - `self.layer_2`: Takes `hidden_size` features (matching the output of `layer_1` after activation) and outputs `output_size` features.
- **`forward` Breakdown:**
  - The input `x` is passed through `layer_1`.
  - The result is passed through `relu`.
  - The activated result is passed through `layer_2` to produce the final output.

## Flexibility of `forward`

While this example shows a simple sequential flow, the `forward` method allows for defining much more complex architectures. You could, for instance, implement:

- Networks with multiple hidden layers.
- Skip connections (where an earlier layer's output is added back in later).
- Branching architectures.

## Summary

Building multi-layer networks with `nn.Module` involves defining the necessary layers (like `nn.Linear`, `nn.ReLU`) as attributes in `__init__` and then explicitly defining the sequence and logic of how data flows through these layers in the `forward` method. Careful attention must be paid to matching the input and output dimensions of consecutive layers.
