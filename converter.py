import torch
import torch.nn as nn
from tqdm import tqdm

class Model(nn.Module):
    def __init__(self,in_features,out_features) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features,out_features)
        self.fc2 = nn.Linear(out_features,1)
    
    def forward(self,x):
        return self.fc2(self.fc1(x))

def sprintf_list(name,List,shape):
    string = ""

    string += "int "
    string += name+f"_schema[{len(shape)}] = "
    string += "{"
    string += ','.join([str(round(num,5)) for num in shape])
    string += "};\n"

    string += "float"
    string += " "
    string += name
    string += f"[{len(List)}] "
    string += "= "
    string += "{"
    string += ','.join([str(num) for num in List])
    string += "};\n"
    return string

def print_weight(name,module: nn.Module,file):
    string=sprintf_list(f"{name}_w",
                        module.weight.data.flatten().detach().cpu().numpy().tolist(),
                        module.weight.data.shape)
    print(string,file=file)
    string=sprintf_list(f"{name}_b",
                        module.bias.data.flatten().detach().cpu().numpy().tolist(),
                        module.bias.data.shape)
    print(string,file=file)

def convert_weight(net: nn.Module, path: str):
    hpp = open(path,"wt")
    print("#ifndef SimpleNN\n#define SimpleNN\n",file=hpp)
    for name, module in net.named_children():
        print(name)
        print_weight(name,module,hpp)
    print("\n#endif",file=hpp)
    return

def convert_model(net: nn.Module, path: str):
    return

if __name__=="__main__":
    
    # define path and model
    path = "Network.hpp"
    net = Model(2,4)

    # train the model
    N = 1000
    epoch = 200
    data = torch.randn(N,2) * 10 + 5
    print(data.shape)
    A = torch.tensor([[4.,6.]])
    b = torch.tensor([[1.]])
    y = data @ A.T + b
    print(y.shape)
    loss_func = nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(),lr=1e-4,momentum=0.9)
    for e in tqdm(range(epoch)):
        optimizer.zero_grad()
        pred = net(data)
        loss = loss_func(pred, y)
        loss.backward()
        optimizer.step()
    print("MSE Loss: {}".format(round(loss.item(),6)))

    # convert weight and model into hpp
    convert_weight(net, path)
    convert_model(net, path)