import torch
import torch.nn.functional as F


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = torch.nn.Linear(100, 256)
        self.layer2 = torch.nn.Linear(256, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        return x


if __name__ == "__main__":
    traced_net = torch.jit.trace(Net(), torch.randn(1, 100))
    torch.jit.save(traced_net, "models/net.pt")
