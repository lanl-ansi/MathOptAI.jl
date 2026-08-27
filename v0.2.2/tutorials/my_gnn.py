import torch
import torch_geometric

class MyGNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch_geometric.nn.GCNConv(2, 3)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        return torch.nn.functional.sigmoid(x)
