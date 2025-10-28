import gc
import math
import random
import time

import math
import random
import time
from torch_sparse import index_select, SparseTensor
from scipy.optimize import linear_sum_assignment
import os
from scipy.sparse import lil_matrix, coo_matrix
from torch import nn, optim
from GCNEmbedding import *
from utils import *
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import adjusted_rand_score, accuracy_score, jaccard_score, normalized_mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
os.environ["OMP_NUM_THREADS"] = "3"
os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
color_red = "\033[1;31m"
color_green = "\033[1;32m"
color_reset = "\033[0m"

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

num_fea=2

data = load_data(args.dataset,num_fea)
data.to(device)

merged_mask = data.train_mask+data.test_mask
merged_mask = numpy.array(merged_mask)
total_mask = data.train_mask+data.test_mask
total_mask = torch.tensor(total_mask).to(device)
cal_loss_list = []
kl_time = []
op_time = []



def clustering_accuracy(y_true, y_pred):
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(w.max() - w)

    accuracy = sum(w[row_ind[i], col_ind[i]] for i in range(len(row_ind))) / y_pred.size
    return accuracy




def KL_loss(graph_emb: torch.Tensor,
                 centers: torch.Tensor,
                 eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:

    kl_start = time.time()
    x2 = graph_emb.square().sum(1, keepdim=True)
    y2 = centers.square().sum(1, keepdim=True).t()
    xy = graph_emb @ centers.t()
    dist_sq = x2 + y2 - 2 * xy
    dist = torch.sqrt(torch.relu(dist_sq) + eps)
    sim = 1.0 / (1.0 + dist)
    q_ij = sim / sim.sum(1, keepdim=True)
    f_j = sim.sum(0)
    tmp = q_ij.square() / f_j
    p_ij = tmp / tmp.sum(1, keepdim=True)
    kl = torch.sum(p_ij * (p_ij + eps).log() - p_ij * (q_ij + eps).log())
    kl_time.append(time.time() - kl_start)
    return kl, p_ij


def reconstruction_loss(A, embedding):
    A_hat = F.sigmoid(torch.matmul(embedding, embedding.T))
    r_loss = torch.norm(A - A_hat)
    return r_loss




def train_semi_supervise_clustering_model(data, centers, constraint_matrix):
    model.train()
    out = model(data.x, data.adj_normalized)
    loss = KL_loss(out[total_mask], centers)[0] + pair_constraint_loss(out[total_mask],constraint_matrix)
    op_start = time.time()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    op_time.append(time.time() - op_start)
    return loss, out





def pre_train_model(data):
    model.train()
    out = model(data.x, data.adj_normalized)

    edge_index = data.edge_index
    num_nodes = data.num_nodes

    adj_matrix_sparse = torch.sparse_coo_tensor(
        indices=edge_index,
        values=torch.ones(edge_index.shape[1], device=device),
        size=(num_nodes, num_nodes)
    )

    adj_matrix_dense = adj_matrix_sparse.to_dense()
    pre_loss = reconstruction_loss(adj_matrix_dense, out)
    optimizer.zero_grad()
    pre_loss.backward()
    optimizer.step()

    return pre_loss


@torch.no_grad()
def test_model(data):
    model.eval()
    embeddings = model(data.x,data.adj_normalized)
    return embeddings


def pair_constraint_loss( choose_embedding, constraint_matrix):
    cal_start = time.time()
    constraint_matrix = constraint_matrix.to_torch_sparse_coo_tensor()
    constraint_sparse = constraint_matrix.coalesce().to(device)
    constraint_dense = constraint_sparse.to_dense()
    indices = constraint_sparse.indices()
    values = constraint_sparse.values()

    N = choose_embedding.size(0)

    row, col = indices[0], indices[1]
    diff = choose_embedding[row] - choose_embedding[col]
    norm_sq = torch.sum(diff ** 2, dim=1)
    constraint_loss = torch.sum(values * norm_sq)
    cal_end = time.time()
    cal_loss_list.append(cal_end-cal_start)
    return constraint_loss



def Constraint_matrix():
    regular_simi = data.regular_simi
    train_label = data.train_label
    total_node = data.total_node
    regular_simi = regular_simi[total_mask]
    sim = regular_simi[:,total_mask]


    # x = data.x[merged_mask]
    # cosine_similarity_matrix = cosine_similarity(x.cpu())
    # cosine_similarity_matrix = torch.tensor(cosine_similarity_matrix)
    # sim = cosine_similarity_matrix.to(device)


    # x = data.x[merged_mask]
    # distance_matrix = euclidean_distances(x.cpu())
    # distance_matrix = torch.tensor(distance_matrix)
    # sim = distance_matrix.to(device)

    n = sim.size(0)
    label = torch.as_tensor(train_label, device=device)
    total_node = torch.as_tensor(list(total_node), device=device)

    pos_k = int(n * args.constraint_rate) + 1
    neg_k = int(n * args.constraint_rate)
    _, topk_idx = sim.topk(pos_k, dim=1, largest=True)
    _, bot_idx  = sim.topk(neg_k, dim=1, largest=False)



    arange_n = torch.arange(n, device=device)[:, None]
    top_mask = (topk_idx != arange_n)
    bot_mask = (bot_idx  != arange_n)

    row_top = arange_n.expand(-1, pos_k)[top_mask]
    col_top = topk_idx[top_mask]
    row_bot = arange_n.expand(-1, neg_k)[bot_mask]
    col_bot = bot_idx[bot_mask]

    in_total   = torch.isin(torch.arange(n, device=device), total_node)
    node_mask  = in_total[:, None] & in_total[None, :]
    label_eq   = (label[:, None] == label[None, :])

    top_sparse    = torch.zeros(n, n, dtype=torch.bool, device=device)
    bottom_sparse = torch.zeros(n, n, dtype=torch.bool, device=device)
    top_sparse[row_top, col_top] = True
    bottom_sparse[row_bot, col_bot] = True

    mask_pos = label_eq & node_mask
    mask_neg = (~label_eq) & node_mask
    mask_top = (~node_mask) & top_sparse
    mask_bot = (~node_mask) & bottom_sparse


    def to_coo(torch_mask, val):
        rc = torch_mask.nonzero(as_tuple=False).cpu().numpy()
        return coo_matrix((np.full(rc.shape[0], val, dtype=np.float32),
                          (rc[:, 0], rc[:, 1])),
                          shape=(n, n))

    coo_pos = to_coo(mask_pos,  1)
    coo_neg = to_coo(mask_neg, -1)
    coo_top = to_coo(mask_top,  1)
    coo_bot = to_coo(mask_bot, -1)

    final_coo = (coo_pos + coo_neg + coo_top + coo_bot).tocoo()

    indices = torch.from_numpy(np.vstack([final_coo.row, final_coo.col])).long().to(device)
    values  = torch.from_numpy(final_coo.data).to(device)
    return torch.sparse_coo_tensor(indices, values, (n, n), device=device)












if __name__ == '__main__':

    """#########  Cluster control  ############"""
    cluster_number = 2
    centriods = torch.rand((cluster_number, 128), dtype=torch.float).to(device)
    constraint_matrix = Constraint_matrix()
    constraint_matrix = constraint_matrix.coalesce()
    constraint_matrix = SparseTensor.from_edge_index(
        constraint_matrix.indices(),
        constraint_matrix.values(),
        sparse_sizes= constraint_matrix.shape)
    constraint_matrix = constraint_matrix.to(device)
    del data.regular_simi
    data.regular_simi = None
    gc.collect()
    torch.cuda.empty_cache()
    """#########  training control ###########"""
    pre_train_flag = 1
    train_flag = 1

    '''################  Feature reconstruction #############'''
    data.x = torch.cat([data.x, data.added_fea[:, -1].view(-1, 1)], dim=-1)
    data_label = data.label.cpu().numpy().astype(int)
    print("fea:", data.x)
    """###############   Initial model ##########"""
    if args.type == "Binary":
        model = GCNemb(num_node_features=data.x.shape[1] , output_dim=2, hidden_dim=128, cluster_num=4).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    print(model)

    loss_set = []

    length = data.train_node_number
    unlabel_mask = torch.ones(length, dtype=torch.bool)
    label_data = data.total_node
    label_data.sort()
    unlabel_mask[label_data] = False
    """##############  Sample construction  ###############"""


    kmeans_time = []
    count_loss = float('inf')
    pre_train = time.time()
    if pre_train_flag == 1:
        for i in range(100):
            pre_loss = pre_train_model(data)
            print(f'pretrain loss: {pre_loss}')
            if count_loss > pre_loss:
                torch.save(model.state_dict(), 'pre_train_model.pth')
    pre_time = time.time()-pre_train
    print("pre_time:",pre_time)
    model.load_state_dict(torch.load('pre_train_model.pth'))
    mode = 'kmeans'
    train_start = time.time()
    if train_flag == 1:
        for i in range(100):
            if mode == 'kmeans':
                loss, out = train_semi_supervise_clustering_model(data, centriods, constraint_matrix)
                kmean_satrt = time.time()
                k_input = out.detach().cpu().numpy()
                k_input = k_input[merged_mask]
                kmeans = KMeans(n_clusters=cluster_number)
                kmeans.fit(k_input)
                centriods = torch.tensor(kmeans.cluster_centers_).to(device)
                cluster_results = kmeans.fit_predict(k_input)
                kmeans_time.append(time.time() - kmean_satrt)
                loss_set.append(loss.item())

            train_nmi = normalized_mutual_info_score(data.label.cpu().numpy()[merged_mask],
                                                     cluster_results)
            train_acc = clustering_accuracy(data_label[merged_mask], cluster_results)
            print(f"epoch: {i}, total_loss: {loss}, ACC: {train_acc}, NMI: {train_nmi}")
        train_end = time.time()




        test_start = time.time()
        emd = test_model(data)
        emd = emd.cpu().numpy()

        """############# Downstream task ##############"""

        merged_data = emd[merged_mask]
        merged_label = data.label[merged_mask]
        print("data_number:", len(merged_label))

        kmeans = KMeans(n_clusters=cluster_number)
        predict_labels = kmeans.fit_predict(merged_data)
        cen = kmeans.cluster_centers_
        print(mode)






        print(args.dataset)
        acc = clustering_accuracy(merged_label.cpu().numpy().astype(int), predict_labels)
        nmi = normalized_mutual_info_score(merged_label.cpu().numpy(), predict_labels)
        ari = adjusted_rand_score(merged_label.cpu().numpy(), predict_labels)
        print(f"ACC: {acc}")
        print(f"NMI: {nmi}")
        print(f'ARI: {ari}')
        test_end = time.time()
        train_time = train_end - train_start
        test_time = test_end-test_start
        cal_loss_time = np.array(cal_loss_list)
        op_time = np.array(op_time)
        kl_time = np.array(kl_time)
        kmeans_time = np.array(kmeans_time)
        print("train_time:",train_time)







