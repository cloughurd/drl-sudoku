import torch
import torch.nn.functional as F

def message_passing(nodes, edges, edge_features, message_fn, edge_keep_prob=1.0):
    """
    Pass messages between nodes and sum the incoming messages at each node.
    Implements equation 1 and 2 in the paper, i.e. m_{.j}^t &= \sum_{i \in N(j)} f(h_i^{t-1}, h_j^{t-1})

    :param nodes: (n_nodes, n_features) tensor of node hidden states.
    :param edges: (n_edges, 2) tensor of indices (i, j) indicating an edge from nodes[i] to nodes[j].
    :param edge_features: features for each edge. Set to zero if the edges don't have features.
    :param message_fn: message function, will be called with input of shape (n_edges, 2*n_features + edge_features). The output shape is (n_edges, n_outputs), where you decide the size of n_outputs
    :param edge_keep_prob: The probability by which edges are kept. Basically dropout for edges. Not used in the paper.
    :return: (n_nodes, n_output) Sum of messages arriving at each node.
    """
    n_nodes = nodes.size(0)
    n_features = nodes.size(1)
    n_edges = edges.size(0)

    message_inputs = torch.gather(nodes, edges).view(-1, 2 * n_features)
    reshaped = torch.concat((message_inputs, edge_features))
    messages = message_fn(reshaped)
    messages = F.dropout(messages, 1 - edge_keep_prob)

    n_output = messages.size(1)
    