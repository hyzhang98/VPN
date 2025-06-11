import torch 


class SimpleNN3(torch.nn.Module):
    def __init__(self, input_channel, input_dim, n_class, device=None):
        super(SimpleNN3, self).__init__()
        self.device = torch.device('cpu') if device is None else device
        self.input_channel = input_channel
        self.input_dim = input_dim
        self.n_class = n_class
        self.dim1 = input_channel * input_dim * input_dim
        self._build_up()
        self.to(self.device)

    def _build_up(self):
        self.fc1 = torch.nn.Linear(self.dim1, 1024)
        self.fc2 = torch.nn.Linear(1024, 1024)
        self.fc3 = torch.nn.Linear(1024, self.n_class)

    def forward(self, x):
        # No dropout!
        x = x.reshape(x.size(0), self.dim1)
        output = self.fc1(x)
        output = output.relu()
        output = self.fc2(output)
        output = output.relu()
        # output = torch.nn.functional.dropout(output, 0.5)
        output = self.fc3(output)
        return output

class SoftmaxRegression(torch.nn.Module):

    def __init__(self, input_channel, input_dim, n_class, device=None):
        super(SoftmaxRegression, self).__init__()
        self.flatten = torch.nn.Flatten()          
        self.linear = torch.nn.Linear(input_channel*input_dim*input_dim, n_class)  
        self.device = torch.device('cpu') if device is None else device
        self.to(self.device)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        return x
