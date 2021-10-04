import torch
from torch import nn
from torch_geometric.data import Data
import torch.nn.functional as F


class DMPNNConv(nn.Module):
    def __init__(self, emb_dim):
        """

        :param emb_dim:  int,
            The embedding dimension for the edge and node feature
        """
        super(DMPNNConv, self).__init__()
        self.W_i = torch.nn.Linear(emb_dim * 2, emb_dim)
        self.W_m = torch.nn.Linear(emb_dim, emb_dim)

    def forward(self, x, h_e, bond_n):
        bond_n = bond_n.view((bond_n.shape[0], bond_n.shape[1], 1))
        h_n = F.relu(self.W_i(torch.cat([x, h_e], dim=1)))
        m = torch.sum(bond_n * h_n, dim=1).sum(dim=0)
        h = F.relu(h_n + self.W_m(m))
        return h


class DMPNNEncoder(nn.Module):
    def __init__(self, num_layers, drop_ratio, JK, emb_dim, num_atom_type, num_bond_type):
        """

        :param num_layers:
        :param drop_ratio:
        :param JK:
        :param emb_dim:
        :param num_atom_type:
        :param num_bond_type:
        """
        super(DMPNNEncoder, self).__init__()
        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK

        # define the embeddings of the atoms and bonds and initilize the embeddings.

        self.atom_embeddings = nn.Embedding(num_atom_type, emb_dim)
        self.bond_embeddings = nn.Embedding(num_bond_type, emb_dim)
        nn.init.xavier_uniform_(self.atom_embeddings.weight.data)
        nn.init.xavier_uniform_(self.bond_embeddings.weight.data)

        # Define the graph convolutional layer of the GNN
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.gnns.append(DMPNNConv(emb_dim))

        # Define the graph normalization layer of the GNN
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

    def forward(self, *argv):
        x, edge_n, bond_n = argv.x, argv.edge_attr, argv.edge_n
        # Get the embeddings for the graph: node feature and edge feature.
        x = self.atom_embeddings(x)
        x = x[:, 0] + x[:, 1]
        edge_n = self.bond_embeddings(edge_n)
        edge_n = edge_n[:, 0] + edge_n[:, 1]

        for i in range(self.num_layers):
            h = self.gnns[i](x, edge_n, bond_n)

        return h


if __name__ == '__main__':
    # x = torch.tensor([[0], [1], [2]], dtype=torch.float)
    # x_edge = torch.tensor([[0], [1], [1], [2]], dtype=torch.float)
    # edge_index = torch.tensor([[0, 1], [1, 0], [1, 2], [2, 1]], dtype=torch.long)
    # edge_attr = torch.tensor([[0, 1], [0, 1], [1, 0], [1, 0]])
    # edge_n = torch.tensor([[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]])
    # data = Data(x=x, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr, edge_n=edge_n)
    # dmpnn = DMPNNConv(emb_dim=512, bond_dim=2, x_dim=1)
    # dmpnn(x_edge, edge_attr, edge_n)
    num_layers = 4
    dropout_ratio = 0.5
    JK = 'concat'
    num_emb_dim = 512
    num_atom_type = 3
    num_bond_type = 2

    x = torch.tensor([[0], [1], [2]], dtype=torch.long)
    x_edge = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.long)
    edge_index = torch.tensor([[0, 1], [1, 0], [1, 2], [2, 1]], dtype=torch.long)
    edge_attr = torch.tensor([[0, 1], [0, 1], [1, 0], [1, 0]], dtype=torch.long)
    edge_n = torch.tensor([[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr, edge_n=edge_n)

    emb1 = nn.Embedding(num_atom_type, num_emb_dim)
    nn.init.xavier_uniform_(emb1.weight.data)
    x_edge = emb1(x_edge)
    emb2 = nn.Embedding(num_bond_type, num_emb_dim)
    nn.init.xavier_uniform_(emb2.weight.data)
    edge_attr = emb2(edge_attr)

    d_mpnn_encoder = DMPNNEncoder(num_layers, dropout_ratio, JK, num_emb_dim, num_atom_type, num_bond_type)
    t = d_mpnn_encoder(data)
