import numpy as np

import torch
import torch.nn
import torch.nn.functional

device = torch.device('cpu')


class MaskedLinear(torch.nn.Linear):
    """
    A linear mapping with a given mask on the weights (arbitrary bias)

    :param in_features: the number of input features
    :type in_features: int
    :param out_features: the number of output features
    :type out_features: int
    :param mask: the mask to apply to the in_features x out_features weight matrix
    :type mask: torch.Tensor
    :param bias: whether or not `MaskedLinear` should include a bias term. defaults to `True`
    :type bias: bool
    """

    def __init__(self, in_features, out_features, mask, bias=True):
        super(MaskedLinear, self).__init__(in_features, out_features, bias)
        self.register_buffer('mask', mask.data)

    def forward(self, _input):
        """
        the forward method that does the masked linear computation and returns the result
        """
        masked_weight = self.weight * self.mask
        return torch.nn.functional.linear(_input, masked_weight, self.bias)


mask = torch.zeros(100, 10)
for i in range(10):
    for j in range(10):
        mask[j * 10 + i][i] = 1

model = torch.nn.Sequential(
    MaskedLinear(10, 100, mask),
    # torch.nn.Linear(10, 100),
    torch.nn.ReLU(),
    torch.nn.Linear(100, 10),
).to(device)

x = torch.zeros(1, 10)
y = torch.zeros(1, 10)

for i in range(10):
    x[0][i] = i
    y[0][i] = i

loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(500):
    # Forward pass: compute predicted y by passing x to the model.
    y_pred = model(x)

    # Compute and print loss.
    loss = loss_fn(y_pred, y)
    print(t, loss.item())

    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable
    # weights of the model). This is because by default, gradients are
    # accumulated in buffers( i.e, not overwritten) whenever .backward()
    # is called. Checkout docs of torch.autograd.backward for more details.
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model
    # parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()
