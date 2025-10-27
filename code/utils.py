import pickle as pkl

import pandas as pd
from torch_geometric import utils
from torch_geometric.utils import from_networkx

from config import *
import random
import numpy
import numpy as np, sys
from torch_geometric.data import Data

# pickle 是python自带的模块。它可以将对象序列化，即以二进制文本的形式将内存中的对象写入一个文件，从而永久保存；也可以反向序列化对象，即从保存对象的文本文件读取并构造对象
# networkx 网络数据结构操作、分析模块


def load_data(dataset_str,num_fea):
    # print("num_fea:", num_fea)
    print("rate", args.rate)
    print("constraint_rate", args.constraint_rate)

    names = [args.feature_type, 'graph', 'train_mask', 'test_mask', 'labeled', 'regular_simi']

    objects = []
    print("开始读数据")
    for i in range(len(names)):
        with open("../data/{}/{}-{}.{}".format(dataset_str, dataset_str, names[i], names[i]),
                  'rb') as f:  # 读取 data/ind.{dataset_str}.{names[i]}的数据集
            if sys.version_info > (3, 0):  # 判断python是否为3.0以上（包括3.0）
                # 使用pickle模块加载数据集对象
                objects.append(pkl.load(f, encoding='latin1'))  # 读取文件(类似字典)
            else:
                objects.append(pkl.load(f))

    feature, graph, train_mask, test_mask, label, regular_simi = tuple(objects)
    print("读取数据完成")
    label = label.astype(np.int64)

    # edge_index = pd.read_pickle(r'../../data/twitch_gamers/twitch_gamers-edge.edge')
    # edge_index = torch.tensor(np.array(edge_index))



    # x = torch.tensor(np.array(feature), dtype=torch.float32)
    label = torch.tensor(np.array(label), dtype=torch.float32)
    num_nodes = len(graph.nodes())


    data = utils.from_networkx(graph)
    edge_index = data.edge_index
    data.x = torch.tensor(np.array(feature), dtype=torch.float32)[:, :num_fea]
    print("data对象构建完成")


    # data.CGCN_x = CGCN_feature
    data.train_mask = train_mask
    data.test_mask = test_mask
    data.label = label
    data.regular_simi = regular_simi
    data.added_fea = torch.tensor(np.array(graph.degree()))
    # degrees = utils.degree(edge_index[0], num_nodes)
    # data.added_fea = degrees
    # update
    merged_mask = data.train_mask + data.test_mask
    merged_mask = numpy.array(merged_mask)
    train_label = label[merged_mask]
    train_labels_0_index = np.where(train_label.cpu().numpy() == 0)[0]
    train_labels_1_index = np.where(train_label.cpu().numpy() == 1)[0]
    train_node_number = len(train_labels_0_index) + len(train_labels_1_index)
    print("开始选节点给标签")
    label_0_node, label_1_node = get_label_node_based_on_influence_score(dataset_str, merged_mask, train_label)
    print("选节点结束")

    data.label_0_node = label_0_node
    data.label_1_node = label_1_node
    data.train_node_number = train_node_number
    # data.own_label_flag = own_label_flag
    total_node = label_0_node + label_1_node
    data.total_node = np.array(total_node)
    data.train_label = train_label
    print("length 0f label_0_node", len(label_0_node))
    print("length of label_1_node", len(label_1_node))
    # update
    device = edge_index.device


    # 1. 加自环
    loop_index = torch.arange(num_nodes, device=device).unsqueeze(0).repeat(2, 1)
    edge_index = torch.cat([edge_index, loop_index], dim=1)  # [2, E+V]



    # 2. 构造未归一化的稀疏邻接（值全 1）
    adj = torch.sparse_coo_tensor(
        indices=edge_index,
        values=torch.ones(edge_index.size(1), device=device),
        size=(num_nodes, num_nodes),
        device=device)

    # 3. 计算度向量
    deg = torch.sparse.sum(adj, dim=1).to_dense()  # [N]
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0  # 孤立点

    # 4. 计算归一化系数：每条边 (i,j) 对应 deg_inv_sqrt[i]*deg_inv_sqrt[j]
    row, col = edge_index
    norm_vals = deg_inv_sqrt[row] * deg_inv_sqrt[col]  # [E+V]

    # 5. 重新组装稀疏张量
    normalized_adj_sparse = torch.sparse_coo_tensor(
        indices=edge_index,
        values=norm_vals,
        size=(num_nodes, num_nodes),
        device=device)

    # 如果后续还想用稀疏矩阵乘法，可直接把 normalized_adj_sparse 传下去
    data.adj_normalized = normalized_adj_sparse
    print(args.dataset)
    print(args.rate)
    print(data)
    return data


# Kendall algorithm
def kendall(a, b):
    Lens = len(a)
    ties_onlyin_x = 0
    ties_onlyin_y = 0
    con_pair = 0
    dis_pair = 0
    for i in range(Lens - 1):
        for j in range(i + 1, Lens):
            test_tying_x = np.sign(a[i] - a[j])
            test_tying_y = np.sign(b[i] - b[j])
            panduan = test_tying_x * test_tying_y
            if panduan == 1:
                con_pair += 1
            elif panduan == -1:
                dis_pair += 1
    Kendallta1 = (2 * (con_pair - dis_pair)) / (len(a) * (len(a) - 1))
    return Kendallta1


def get_rank(data):
    ranked_data = sorted(data)[::-1]
    rank = []
    for o_num in data:
        for r_num in ranked_data:
            if o_num == r_num:
                rank.append(ranked_data.index(o_num) + 1)
                break
    return rank


def get_label_node_based_on_influence_score(dataset, merged_mask, train_label):
    # 指定文件路径
    file_path = "../data/influ_score/{}-graph_SIR.txt".format(dataset)  # 修改为你的文件名和路径

    # 用于保存第二列的数据
    second_column = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for index, line in enumerate(file):
            parts = line.strip().split()  # 对行进行分割，默认按空格分割
            if len(parts) >= 2:
                second_column.append(parts[1])  # 保存索引和第二列值

    second_column = np.array(second_column)
    second_column = second_column[merged_mask]
    indices = np.arange(second_column.size)
    second_column_with_index = np.array(list(zip(indices, second_column)))
    # 按照第二列的值从大到小排序
    sorted_column_with_index = sorted(second_column_with_index, key=lambda x: float(x[1]), reverse=True)

    node_number = [int(index) for index, _ in sorted_column_with_index]
    train_label = train_label.numpy()
    node_number_label = train_label[node_number]
    node_number = np.array(node_number)
    label_1_node = node_number[:int(len(node_number) * args.rate * (1 / 3))]
    label_0_node = node_number[-int(len(node_number) * args.rate * (2 / 3)):]
    return label_0_node.tolist(), label_1_node.tolist()
