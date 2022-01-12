import torch as t
import numpy as np
import matplotlib.pyplot as plt


def make_dataset(x,y, device = t.device('cpu')):
    x_transformed = x.astype('float32').reshape(-1,28*28)/255.0
    x_t = t.from_numpy(x_transformed)
    y_t = t.from_numpy(y.astype('long'))
    return t.utils.data.TensorDataset(x_t.to(device), y_t.to(device))


def make_model(width, drop):
    return t.nn.Sequential(t.nn.Linear(28*28, 2*width), t.nn.Dropout(drop), t.nn.ReLU(),
                   t.nn.Linear(2*width, width), t.nn.Dropout(drop),  t.nn.ReLU(),
                   t.nn.Linear(width, width),t.nn.Dropout(drop),  t.nn.ReLU(),
                   t.nn.Linear(width, 10)
                   )


def init_layer(layer):
    if isinstance(layer,t.nn.modules.linear.Linear):
        fan_in = layer.weight.size(1)
        sigma = 1*np.sqrt(6/fan_in)
        t.nn.init.uniform_(layer.weight,-sigma,sigma)
        if layer.bias is not None:
            t.nn.init.zeros_(layer.bias)

def init_layer_with_sigma(sigma=1):
    def f(layer):
        if isinstance(layer,t.nn.modules.linear.Linear):
            fan_in = layer.weight.size(1)
            s = sigma*np.sqrt(6/fan_in)
            t.nn.init.uniform_(layer.weight, -s , s)
            if layer.bias is not None:
                t.nn.init.zeros_(layer.bias)
    return f


def accuracy(model, inp, target):
    pred = t.softmax(model(inp),dim=1)
    pred_class = t.argmax(pred,dim=1)
    return t.sum(pred_class == target).item()/len(target)

def prediction(model, img):
    pred = model(img)
    p = t.softmax(pred,1)
    l = t.argmax(p)
    return (l.item(), p[0][l].item())

def model_detach(model):
    for p in model.parameters():
        p.requires_grad = False

def model_atach(model):
    for p in model.parameters():
        p.requires_grad = True

def display(d):
    return plt.imshow(d.numpy().reshape(28,28),cmap="Greys")

def numel(model):
    n=0
    for p in model.parameters():
        n+=p.numel()
    
    return n 
