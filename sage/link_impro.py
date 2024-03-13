import torch
from torch_geometric.nn import SAGEConv
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import matplotlib.pyplot as plt

import torch.nn.functional as F
import torch_geometric.transforms as T

from torch_geometric.utils import negative_sampling
from torch_geometric.utils import to_undirected

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, 1))  

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
        return torch.sigmoid(x)

dataset = PygNodePropPredDataset(name='ogbn-arxiv', transform=T.ToSparseTensor())
data = dataset[0]

data.adj_t = data.adj_t.to_symmetric()

model = SAGE(data.num_features, 256, 3, 0.5).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)  

split_idx = dataset.get_idx_split()
train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

def train():
    model.train()
    optimizer.zero_grad()

    out = model(data.x, data.adj_t)

    pos_weight = torch.tensor([data.y[train_idx].size(0) / torch.sum(data.y[train_idx])])
    pos_loss = F.binary_cross_entropy_with_logits(out[train_idx].squeeze(), data.y[train_idx].squeeze().float(), pos_weight=pos_weight.to(device))
    
    edge_index_tuple = data.adj_t.coo()[:2]
    edge_index = torch.stack(edge_index_tuple, dim=0).to(device)
    data.edge_index = to_undirected(edge_index)
    
    neg_edge_index = negative_sampling(data.edge_index, num_neg_samples=data.edge_index.size(1))
    neg_out = model(data.x, neg_edge_index)
    neg_loss = F.binary_cross_entropy(neg_out.squeeze(), torch.zeros(neg_out.size(0)).to(device))
    
    loss = pos_loss + neg_loss
    loss.backward()
    optimizer.step()

    return loss.item()

# def evaluate():
#     model.eval()
#     with torch.no_grad():
#         out = model(data.x, data.adj_t)
#         y_true = data.y.bool()
#         y_pred = out > 0.5
#         pos_acc = (y_pred == y_true).sum().item() / y_true.size(0)
        
#         neg_edge_index = negative_sampling(data.edge_index, num_neg_samples=data.edge_index.size(1))
#         neg_out = model(data.x, neg_edge_index)
#         neg_acc = torch.lt(neg_out, 0.5).sum().item() / neg_out.size(0)
        
#         return pos_acc, neg_acc

def evaluate():
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.adj_t)
        y_true = data.y.bool()
        y_pred = out > 0.5
        pos_correct = (y_pred & y_true).sum().item()
        pos_total = y_true.sum().item()

        neg_edge_index = negative_sampling(data.edge_index, num_neg_samples=data.edge_index.size(1))
        neg_out = model(data.x, neg_edge_index)
        neg_correct = torch.lt(neg_out, 0.5).sum().item()
        neg_total = neg_out.size(0)

        return pos_correct, neg_correct, pos_total, neg_total

loss_values = []
pos_acc_values = []
neg_acc_values = []
total_acc_values = []

# for epoch in range(1000):
#     loss = train()
#     pos_acc, neg_acc = evaluate()
#     scheduler.step()
#     total_acc = (pos_acc + neg_acc) / 2  
#     print(f'Epoch: {epoch+1}, Loss: {loss:.4f}, Pos Accuracy: {pos_acc:.4f}, Neg Accuracy: {neg_acc:.4f}, Total Accuracy: {total_acc:.4f}')

#     loss_values.append(loss)
#     pos_acc_values.append(pos_acc)
#     neg_acc_values.append(neg_acc)
#     total_acc_values.append(total_acc)  

for epoch in range(1000):
    loss = train()
    pos_correct, neg_correct, pos_total, neg_total = evaluate()
    scheduler.step()
    pos_acc = pos_correct / pos_total
    neg_acc = neg_correct / neg_total
    total_acc = (pos_correct + neg_correct) / (pos_total + neg_total)
    print(f'Epoch: {epoch+1}, Loss: {loss:.4f}, Pos Accuracy: {pos_acc:.4f}, Neg Accuracy: {neg_acc:.4f}, Total Accuracy: {total_acc:.4f}')

    loss_values.append(loss)
    pos_acc_values.append(pos_acc)
    neg_acc_values.append(neg_acc)
    total_acc_values.append(total_acc)

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Train Loss', color=color)
ax1.plot(loss_values, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Accuracy', color=color)
ax2.plot(pos_acc_values, color=color, label='Pos Accuracy')
ax2.plot(neg_acc_values, color='tab:green', label='Neg Accuracy')
ax2.plot(total_acc_values, color='tab:orange', label='Total Accuracy')  
ax2.tick_params(axis='y', labelcolor=color)
ax2.legend(loc='upper left')  

fig.tight_layout()
plt.savefig(f'loss_acc_plot_{epoch + 1}.png')
plt.show()