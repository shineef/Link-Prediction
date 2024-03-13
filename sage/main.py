import torch
from torch_geometric.nn import SAGEConv
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import matplotlib.pyplot as plt

import torch.nn.functional as F
import torch_geometric.transforms as T

from torch_geometric.utils import negative_sampling


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# class GraphSAGE(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
#         super(GraphSAGE, self).__init__()
#         self.conv1 = SAGEConv(in_channels, hidden_channels)
#         self.conv2 = SAGEConv(hidden_channels, out_channels)
#         self.bn = torch.nn.BatchNorm1d(hidden_channels)
#         self.dropout = torch.nn.Dropout(dropout)

#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index)
#         x = self.bn(x)
#         x = x.relu()
#         x = self.dropout(x)
#         x = self.conv2(x, edge_index)
#         return x.log_softmax(dim=-1)

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)

# dataset = PygNodePropPredDataset(name='ogbn-arxiv')
dataset = PygNodePropPredDataset(name='ogbn-arxiv', transform=T.ToSparseTensor())
data = dataset[0]

data.adj_t = data.adj_t.to_symmetric()

# model = GraphSAGE(dataset.num_features, 256, dataset.num_classes).to(device)
model = SAGE(data.num_features, 256, dataset.num_classes, 3, 0.5).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

split_idx = dataset.get_idx_split()
train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

# def train():
#     model.train()
#     optimizer.zero_grad()
#     out = model(data.x, data.edge_index)
#     loss = torch.nn.functional.nll_loss(out[train_idx], data.y[train_idx].squeeze())
#     loss.backward()
#     optimizer.step()
#     return loss.item()

def train():
    model.train()
    optimizer.zero_grad()

    out = model(data.x, data.adj_t)
    loss = torch.nn.functional.nll_loss(out[train_idx], data.y[train_idx].squeeze())
    loss.backward()
    optimizer.step()
    return loss.item()

evaluator = Evaluator(name='ogbn-arxiv')

# def evaluate():
#     model.eval()
#     with torch.no_grad():  
#         out = model(data.x, data.edge_index)
#         y_pred = out.argmax(dim=-1, keepdim=True)  

#         test_y_pred = y_pred[test_idx]
#         test_y_true = data.y[test_idx]  

#         input_dict = {"y_true": test_y_true, "y_pred": test_y_pred}

#         result = evaluator.eval(input_dict)
#         return result['acc']

def evaluate():
    model.eval()
    with torch.no_grad():

        out = model(data.x, data.adj_t)
        y_pred = out.argmax(dim=-1, keepdim=True)

        test_y_pred = y_pred[test_idx]
        test_y_true = data.y[test_idx]

        input_dict = {"y_true": test_y_true, "y_pred": test_y_pred}

        result = evaluator.eval(input_dict)
        return result['acc']

loss_values = []
acc_values = []

for epoch in range(500):
    loss = train()
    acc = evaluate()
    print(f'Epoch: {epoch+1}, Loss: {loss:.4f}, Accuracy: {acc:.4f}')

    loss_values.append(loss)
    acc_values.append(acc)

fig, ax1 = plt.subplots()

color = 'tab:red'

ax1.set_xlabel('Epoch')
ax1.set_ylabel('Train Loss', color=color)
ax1.plot(loss_values, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Evaluation Accuracy', color=color)
ax2.plot(acc_values, color=color)
ax2.tick_params(axis='y', labelcolor=color)
plt.savefig('loss_acc_plot_single.png')
plt.show()