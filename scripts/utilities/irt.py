import torch

def one_parameter_irt(theta, beta):
    pred = theta - beta
    pred = torch.sigmoid(torch.tensor(pred))
    return pred
