import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()

        self.input_layer = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.input_layer(x)
        return torch.log_softmax(x, dim=1)

    def predict(self, x):
        return self.forward(torch.Tensor(x)).detach().numpy().argmax(axis=1)
