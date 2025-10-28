import pickle as pkl
from torch_geometric import utils
from config import *
import numpy
import numpy as np, sys





def load_data(dataset_str,num_fea):
    print("rate", args.rate)
    print("constraint_rate", args.constraint_rate)
    names = [args.feature_type, 'graph', 'train_mask', 'test_mask', 'labeled', 'regular_simi']
    objects = []
    for i in range(len(names)):
        with open("../data/{}/{}-{}.{}".format(dataset_str, dataset_str, names[i], names[i]),
                  'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    feature, graph, train_mask, test_mask, label, regular_simi = tuple(objects)
    label = label.astype(np.int64)
    label = torch.tensor(np.array(label), dtype=torch.float32)
    num_nodes = len(graph.nodes())
    data = utils.from_networkx(graph)
    edge_index = data.edge_index
    data.x = torch.tensor(np.array(feature), dtype=torch.float32)[:, :num_fea]
    data.train_mask = train_mask
    data.test_mask = test_mask
    data.label = label
    data.regular_simi = regular_simi
    data.added_fea = torch.tensor(np.array(graph.degree()))
    merged_mask = data.train_mask + data.test_mask
    merged_mask = numpy.array(merged_mask)
    train_label = label[merged_mask]
    train_labels_0_index = np.where(train_label.cpu().numpy() == 0)[0]
    train_labels_1_index = np.where(train_label.cpu().numpy() == 1)[0]
    train_node_number = len(train_labels_0_index) + len(train_labels_1_index)
    label_0_node, label_1_node = get_label_node_based_on_influence_score(dataset_str, merged_mask, train_label)
    data.label_0_node = label_0_node
    data.label_1_node = label_1_node
    data.train_node_number = train_node_number
    total_node = label_0_node + label_1_node
    data.total_node = np.array(total_node)
    data.train_label = train_label
    device = edge_index.device

    loop_index = torch.arange(num_nodes, device=device).unsqueeze(0).repeat(2, 1)
    edge_index = torch.cat([edge_index, loop_index], dim=1)

    adj = torch.sparse_coo_tensor(
        indices=edge_index,
        values=torch.ones(edge_index.size(1), device=device),
        size=(num_nodes, num_nodes),
        device=device)
    deg = torch.sparse.sum(adj, dim=1).to_dense()
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    row, col = edge_index
    norm_vals = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    normalized_adj_sparse = torch.sparse_coo_tensor(
        indices=edge_index,
        values=norm_vals,
        size=(num_nodes, num_nodes),
        device=device)
    data.adj_normalized = normalized_adj_sparse
    print(args.dataset)
    print(args.rate)
    print(data)
    return data


def get_label_node_based_on_influence_score(dataset, merged_mask, train_label):
    file_path = "../data/influ_score/{}-graph_SIR.txt".format(dataset)
    second_column = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for index, line in enumerate(file):
            parts = line.strip().split()
            if len(parts) >= 2:
                second_column.append(parts[1])
    second_column = np.array(second_column)
    second_column = second_column[merged_mask]
    indices = np.arange(second_column.size)
    second_column_with_index = np.array(list(zip(indices, second_column)))
    sorted_column_with_index = sorted(second_column_with_index, key=lambda x: float(x[1]), reverse=True)
    node_number = [int(index) for index, _ in sorted_column_with_index]
    train_label = train_label.numpy()
    node_number_label = train_label[node_number]
    node_number = np.array(node_number)
    label_1_node = node_number[:int(len(node_number) * args.rate * (1 / 3))]
    label_0_node = node_number[-int(len(node_number) * args.rate * (2 / 3)):]
    return label_0_node.tolist(), label_1_node.tolist()
