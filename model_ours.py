import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv, GATConv, GCNConv


class GAEBranch(nn.Module):
    def __init__(self, n_nodes, n_embed, n_hidden, backbone="GCN", cheb_k=2):
        super(GAEBranch, self).__init__()
        self.embedding = nn.Embedding(n_nodes, n_embed)
        self.backbone = backbone.upper()

        if self.backbone == "GCN":
            self.conv1 = GCNConv(n_embed, n_hidden)
            self.conv2 = GCNConv(n_hidden, n_hidden)
        elif self.backbone == "GAT":
            self.conv1 = GATConv(n_embed, n_hidden, heads=1, concat=False)
            self.conv2 = GATConv(n_hidden, n_hidden, heads=1, concat=False)
        elif self.backbone == "CHEB":
            self.conv1 = ChebConv(n_embed, n_hidden, K=cheb_k)
            self.conv2 = ChebConv(n_hidden, n_hidden, K=cheb_k)
        else:
            raise NotImplementedError(f"Unsupported GAE backbone: {backbone}")

    def reset_parameters(self):
        self.embedding.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def encode(self, prop_edge_index, args):
        x = self.embedding.weight
        x = F.relu(self.conv1(x, prop_edge_index))
        x = F.dropout(x, p=args.dropout, training=args.training)
        z = self.conv2(x, prop_edge_index)
        return z

    def decode_topology(self, z, edge_index, sigmoid=True):
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value


class ContextBranch(nn.Module):
    def __init__(self, n_nodes, n_embed, n_hidden):
        super(ContextBranch, self).__init__()
        self.embedding = nn.Embedding(n_nodes, n_embed)
        self.context_mapper = nn.Sequential(
            nn.Linear(n_embed, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
        )

    def reset_parameters(self):
        self.embedding.reset_parameters()
        for layer in self.context_mapper:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

    def encode_context(self, args):
        x = self.embedding.weight
        x = self.context_mapper[0](x)
        x = self.context_mapper[1](x)
        x = F.dropout(x, p=args.dropout, training=args.training)
        z = self.context_mapper[2](x)
        return z


class EdgeClassifier(nn.Module):
    def __init__(self, n_hidden, n_edge_attr, num_classes, dropout=0.5):
        super(EdgeClassifier, self).__init__()
        self.edge_attr_lin = nn.Linear(n_edge_attr, n_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.classifier_1 = nn.Linear(n_hidden * 3, n_hidden)
        self.classifier_2 = nn.Linear(n_hidden, num_classes)

    def reset_parameters(self):
        self.edge_attr_lin.reset_parameters()
        self.classifier_1.reset_parameters()
        self.classifier_2.reset_parameters()

    def extract_hidden(self, edge_node_features, edge_attr):
        edge_attr_emb = self.relu(self.edge_attr_lin(edge_attr))
        edge_emb = torch.cat([edge_node_features, edge_attr_emb], dim=1)
        return self.dropout(F.relu(self.classifier_1(edge_emb)))

    def forward(self, edge_node_features, edge_attr):
        edge_hidden = self.extract_hidden(edge_node_features, edge_attr)
        return self.classifier_2(edge_hidden)
