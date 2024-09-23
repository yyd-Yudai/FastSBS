import numpy as np
from numba import jit, int64, float64, prange


@jit('float64(float64[:], float64[:])', nopython=True, cache=True)
def distance(x, y):
    d = 0.0

    for i in range(x.shape[0]):
        diff = x[i] - y[i]
        d += diff * diff

    return np.sqrt(d)


@jit('int64[:](float64[:,:], int64, int64, int64)', nopython=True, cache=True)
def farthest_point_sampling(x, n, num_clusters, seed=0):
    np.random.seed(seed)
    centroids_index = np.zeros(num_clusters, dtype=np.int64)
    centroids_index[0] = np.random.randint(n)
    min_distances = np.full(n, np.inf)

    for i in range(1, num_clusters):
        last_centroid = x[centroids_index[i-1]]

        for j in range(n):
            dist_to_last_centroid = distance(x[j], last_centroid)

            if dist_to_last_centroid < min_distances[j]:
                min_distances[j] = dist_to_last_centroid

        max_dist_index = np.argmax(min_distances)

        while max_dist_index in centroids_index:
            min_distances[max_dist_index] = -np.inf
            max_dist_index = np.argmax(min_distances)
        
        centroids_index[i] = max_dist_index
    
    return centroids_index


@jit('int64[:](float64[:,:], int64, int64, int64[:])', nopython=True, cache=True)
def grouping(x, n, num_clusters, centroids_index):
    groups = np.zeros(n, dtype=np.int64)

    for i in range(n):
        dist_to_centroids = np.zeros(num_clusters, dtype=np.float64)

        for j in range(num_clusters):
            dist_to_centroids[j] = distance(x[i], x[centroids_index[j]])
        
        groups[i] = np.argmin(dist_to_centroids)
    
    return groups


@jit('float64[:](float64[:,:])', nopython=True, cache=True)
def calc_centroid(x):
    n = x.shape[0]

    return np.sum(x, axis=0) / n



@jit('Tuple((float64[:,:], int64[:]))(float64[:,:], int64, int64, int64, int64)', nopython=True, cache=True)
def build_tree(x, n, num_clusters, max_depth, seed=0):
    leaf_nodes_parent = np.zeros(n, dtype=np.int64)
    in_nodes_parent = [-1]
    in_nodes_pos = np.zeros((n, x.shape[1]), dtype=np.float64)
    in_nodes_pos[0] = calc_centroid(x)
    current_node = 1
    nodes_per_depth = [[-1] for _ in range(max_depth)]

    for depth in range(max_depth):

        if depth == 0:
            sampled_index = farthest_point_sampling(x, n, num_clusters, seed)
            groups = grouping(x, n, num_clusters, sampled_index)

            for i in range(num_clusters):
                subset_index = np.where(groups == i)[0]
                subset = x[subset_index]

                in_nodes_pos[current_node] = calc_centroid(subset)
                in_nodes_parent.append(depth)
                leaf_nodes_parent[subset_index] = current_node
                nodes_per_depth[depth].append(current_node)
                current_node += 1
        else:

            for i in range(1, len(nodes_per_depth[depth-1])):
                y_index = np.where(leaf_nodes_parent == nodes_per_depth[depth-1][i])[0]
                y_size = len(y_index)

                if y_size > num_clusters:
                    y = x[y_index]
                    sampled_index = farthest_point_sampling(y, y_size, num_clusters, seed)
                    groups = grouping(y, y_size, num_clusters, sampled_index)

                    for j in range(num_clusters):
                        subset_index = np.where(groups == j)[0]

                        if len(subset_index) != 0:
                            subset = y[subset_index]
                        
                            in_nodes_pos[current_node] = calc_centroid(subset)
                            in_nodes_parent.append(nodes_per_depth[depth-1][i])
                            leaf_nodes_parent[y_index[subset_index]] = current_node
                            nodes_per_depth[depth].append(current_node)
                            current_node += 1
    
    all_nodes_parent = np.zeros(n + current_node, dtype=np.int64)
    all_nodes_parent[:current_node] = np.array(in_nodes_parent)
    all_nodes_parent[current_node:] = leaf_nodes_parent

    all_nodes_pos = np.zeros((n + current_node, x.shape[1]), dtype=np.float64)
    all_nodes_pos[:current_node] = in_nodes_pos[:current_node]
    all_nodes_pos[current_node:] = x

    return all_nodes_pos, all_nodes_parent
                            

@jit('float64[:](float64[:,:], int64[:])', nopython=True, cache=True, parallel=True)
def calc_edge_weight(all_nodes_pos, all_nodes_parent):
    n = len(all_nodes_parent)
    all_edges_weight = np.zeros(n, dtype=np.float64)

    for i in prange(1, n):
        x = all_nodes_pos[i]
        y = all_nodes_pos[all_nodes_parent[i]]
        all_edges_weight[i] = distance(x, y)

    return all_edges_weight


@jit("Tuple((int64[:], float64[:]))(float64[:,:], int64, int64, int64, int64)", nopython=True, cache=True)
def Tree_Construction(x, n, num_clusters, max_depth, seed=0):
    all_nodes_pos, all_nodes_parent = build_tree(x, n, num_clusters, max_depth, seed)
    all_edges_weight = calc_edge_weight(all_nodes_pos, all_nodes_parent)

    return all_nodes_parent, all_edges_weight


@jit('float64[:](int64, float64[:], int64[:])', nopython=True, cache=True)
def calc_flow_mass(n, a, all_nodes_parent):
    u = np.zeros(len(all_nodes_parent), dtype=np.float64)

    for i in range(n):
        u[-n+i] += a[i]
        current_node_parent = all_nodes_parent[-n+i]
        u[current_node_parent] += u[-n+i]

    for i in range(len(u) - n -1, 0, -1):
        current_node_parent = all_nodes_parent[i]
        u[current_node_parent] += u[i]

    return u


@jit('float64[:](int64, float64[:], float64[:], int64[:], float64[:])', nopython=True, cache=True)
def calc_gradient(n, u, v, all_nodes_parent, all_edges_weight):
    g = np.zeros(len(u), dtype=np.float64)
    sgn = np.sign(u - v)

    for i in range(1, len(u)):
        g[i] += all_edges_weight[i] * sgn[i]
        p = all_nodes_parent[i]
        g[i] += g[p]

    g -= g[-1]

    return g[-n:]


def update_node_weights(a, g, I, V, lr):
    a[I] = a[I] * np.exp(- lr * V * (g[I] - g[I].min()))
    a[I] = V * a[I] / a[I].sum()
    return a


def tree_stealth_sampling_fixed(Z, K, num_clusters, max_depth, n_slice, eps):
    assert len(Z) == len(K)

    C = len(Z)      
    V = np.array(K) 

    X = np.concatenate(Z, axis=0)
    col_max = np.max(X, axis=0)
    X = X / col_max
    X = X + 1e-10 * np.random.randn(*X.shape)
    N = len(X)
    Z_shapes = np.array([arr.shape[0] for arr in Z])
    I = np.split(np.arange(N), np.cumsum(Z_shapes)[:-1])

    b = np.array([np.sum(V) / N for i in range(N)])
    a = np.zeros(N)
    for i in range(C):
        a[I[i]] = V[i] / N

    tree = []
    for i in range(n_slice):
        all_nodes_parent, all_edges_weight = Tree_Construction(X, N, num_clusters, max_depth, seed=i)
        tree.append([all_nodes_parent, all_edges_weight])

    a_cum = np.zeros_like(a)
    a_mean = []
    obj = []
    lr_sum = 0
    # iteration
    for itr in range(30000):
        t = np.random.randint(n_slice)
        all_nodes_parent, all_edges_weight = tree[t]

        u = calc_flow_mass(N, a, all_nodes_parent)
        v = calc_flow_mass(N, b, all_nodes_parent)

        g = calc_gradient(N, u, v, all_nodes_parent, all_edges_weight)

        # learning rate
        lr = 1.0 / np.sqrt(itr + 1) / (3 * np.sum(K))
        lr_sum = lr_sum + lr

        # update
        for i in range(C):
            a = update_node_weights(a, g, I[i], V[i], lr)

        a_cum = a_cum + lr * a
        a_mean.append(a_cum / lr_sum)

        if itr >= 100:
            diff = np.abs(a_mean[-2] - a_mean[-1])
            if np.sum(diff) < eps:
                break

    a_mean[-1] = a_mean[-1] / np.sum(a_mean[-1])
    p = np.split(a_mean[-1], np.cumsum(Z_shapes))

    return [np.array(p[:-1][i]) for i in range(C)], obj



def tree_stealth_sampling_randomized(Z, K, num_clusters, max_depth, eps):
    assert len(Z) == len(K)

    C = len(Z)      
    V = np.array(K) 

    X = np.concatenate(Z, axis=0)
    col_max = np.max(X, axis=0)
    X = X / col_max
    X = X + 1e-10 * np.random.randn(*X.shape)
    N = len(X)
    Z_shapes = np.array([arr.shape[0] for arr in Z])
    I = np.split(np.arange(N), np.cumsum(Z_shapes)[:-1])

    b = np.array([np.sum(V) / N for i in range(N)])
    a = np.zeros(N)
    for i in range(C):
        a[I[i]] = V[i] / N

    a_cum = np.zeros_like(a)
    a_mean = []
    obj = []
    lr_sum = 0
    # iteration
    for itr in range(30000):
        all_nodes_parent, all_edges_weight = Tree_Construction(X, N, num_clusters, max_depth, seed=itr)

        u = calc_flow_mass(N, a, all_nodes_parent)
        v = calc_flow_mass(N, b, all_nodes_parent)

        g = calc_gradient(N, u, v, all_nodes_parent, all_edges_weight)

        # learning rate
        lr = 1.0 / np.sqrt(itr + 1) / (3 * np.sum(K))
        lr_sum = lr_sum + lr

        # update
        for i in range(C):
            a = update_node_weights(a, g, I[i], V[i], lr)

        a_cum = a_cum + lr * a
        a_mean.append(a_cum / lr_sum)

        if itr >= 100:
            diff = np.abs(a_mean[-2] - a_mean[-1])
            if np.sum(diff) < eps:
                break

    a_mean[-1] = a_mean[-1] / np.sum(a_mean[-1])
    p = np.split(a_mean[-1], np.cumsum(Z_shapes))

    return [np.array(p[:-1][i]) for i in range(C)], obj