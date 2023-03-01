import torch
import numpy as np

class ManeuverTimeHead(torch.nn.Module):
    def __init__(self, in_features) -> None:
        super().__init__()
        self.linear1 = torch.nn.Linear(in_features, out_features=5)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(5, out_features=1)
        tanh = torch.nn.Tanh()
        self.output_fn = lambda x : 0.5 * (1.0 + tanh(x)) # torch.nn.Sigmoid()

    def forward(self, embedding):
        x = self.linear1(embedding)
        x = self.relu(x)
        x = self.linear2(x)
        return torch.squeeze(self.output_fn(x))

class DeltaVelocityHead(torch.nn.Module):
    def __init__(self, in_features) -> None:
        super().__init__()
        self.linear1 = torch.nn.Linear(in_features, out_features=5)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(5, out_features=1)
        tanh = torch.nn.Tanh()
        self.output_fn = lambda x : 0.5 * (1.0 + tanh(x)) # torch.nn.Sigmoid()

    def forward(self, embedding):
        x = self.linear1(embedding)
        x = self.relu(x)
        x = self.linear2(x)
        return torch.squeeze(self.output_fn(x)) # Dv max is like 1.5 in absolute value
    
class Wrapper(torch.nn.Module):
    def __init__(self, convnet, model_to_train)  -> None:
        super().__init__()
        self.convnet = convnet
        self.model_to_train = model_to_train
    
    def forward(self, x):
        c, embedding = self.convnet(x) # NOTE : we suppose we only send in data with manoeuver in it
        output = self.model_to_train(embedding) 
        return output, embedding