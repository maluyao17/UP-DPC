import random
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.preprocessing import MinMaxScaler
from Dataprocessing import mapminmax
from Dataprocessing import load_Data, load_Data2, get_data, Euc_dist, normalize_by_minmax
import datetime
# from sklearn.cluster import KMeans
import random
import torch
from fast_pytorch_kmeans import KMeans
import math
from Dataprocessing import get_8m
try:
    import pynvml
    _pynvml_exist = True
except ModuleNotFoundError:
    _pynvml_exist = False
file_Name = ['E:/mnist8m/mnist1q.mat', 'E:/mnist8m/mnist2q.mat',
                'E:/mnist8m/mnist3q.mat','E:/mnist8m/mnist4q.mat',
                'E:/mnist8m/mnist5q.mat','E:/mnist8m/mnist6q.mat',
                'E:/mnist8m/mnist7q.mat','E:/mnist8m/mnist8q.mat',
                'E:/mnist8m/mnist9q.mat','E:/mnist8m/mnist10q.mat',
                'E:/mnist8m/mnist11q.mat','E:/mnist8m/mnist12q.mat',
                'E:/mnist8m/mnist13q.mat','E:/mnist8m/mnist14q.mat',
                'E:/mnist8m/mnist15q.mat','E:/mnist8m/mnist16q.mat']
# file_Name = ['E:/mnist8m/mnist1q.mat', 'E:/mnist8m/mnist2q.mat']

scaler = MinMaxScaler(feature_range=(0, 1))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def random_walk(G, path_length, alpha, rand=random.Random(), start=None):
    """ 
        Pure random walk, using Markov stochastic processes
    """
    if start:
        path = [start]
    else:
        path = [rand.choice(list(G.nodes()))]

    while len(path) < path_length:
        cur = path[-1]
        if G.degree(cur) > 0:
            if rand.random() >= alpha:
                # print(list(G.neighbors(cur)))
                path.append(rand.choice(list(G.neighbors(cur))))
            else:
                path.append(path[0])
        else:
            break
    return [node for node in path]

def random_Walk_ECPCS(similarity_mat, length):
    """
        Python implementation of random walk in 'ECPCS'
        params:
            similarity_mat: similarity matrix
            l: The step size of each random walk
        returns:
            newS:The output similarity matrix
    """
    # starttime = datetime.datetime.now()
    size_S = similarity_mat.shape[0]
    for i in range(0, size_S):
        similarity_mat[i, i] = 0

    rowSum = torch.transpose(torch.sum(similarity_mat, axis=1), -1, 0)
    rowSum = rowSum.reshape(size_S, 1)
    rowSums = torch.tile(rowSum, (1, size_S))
    find_arr = torch.where(rowSums == 0)
    for i in range(0, len(find_arr[0])):
        rowSums[find_arr[0][i]][find_arr[1][i]] = -1.0
    
    P_mat = similarity_mat / rowSums
    find_arr = torch.where(P_mat < 0)
    for i in range(0, len(find_arr[0])):
        P_mat[find_arr[0][i]][find_arr[1][i]] = 0.0

    tempP = P_mat
    inProdP = torch.mm(P_mat, torch.transpose(P_mat, -1, 0))
    for i in range(0, length):
        tempP = torch.mm(tempP, P_mat)
        inProdP = inProdP + torch.mm(tempP, torch.transpose(tempP, -1, 0))
    
    # endtime = datetime.datetime.now()
    # print(endtime - starttime)
    diag_inProdP = torch.diag(inProdP)
    diag_inProdP = diag_inProdP.reshape(len(inProdP), 1)
    inProdii = torch.tile(diag_inProdP, (1, size_S))
    inProdjj = torch.transpose(inProdii, -1, 0)
    newS = inProdP / torch.sqrt(torch.multiply(inProdii, inProdjj))
    sr = torch.sum(torch.transpose(P_mat, -1, 0), axis=0)
    isolatedIdx = torch.stack(torch.where(sr < 10e-10))
    if len(isolatedIdx) > 0:
        newS[isolatedIdx, :] = 0
        newS[:, isolatedIdx] = 0
    # sr = np.sum(np.transpose(P_mat), axis=0)
    return newS

class Dpcpp:

    def __init__(self, label, kms_center, data_to_kms, s, p, l, N, fn=16 ,r=100):
        # self.data = data
        self. label = label
        self.s = s
        self.p = p
        self.l = l
        self.r = r
        # self.k = len(set((label.cpu()).numpy().tolist()))
        self.k = len(set(label.tolist()))
        # self.const_N = N
        self.N = fn*N
        self.cluster = self.cluster = torch.Tensor([-1] * self.p).to(device)
        self.center = [-1] * self.k
        self.sample_label = [-1]*self.N
        self.kmeans_center = kms_center
        self.data_to_kms = data_to_kms
        self.label = []
        # self.index_return = index_return
    
    def fit(self):
        similarity_mat, Z = self.sample_Data()
        start_time = datetime.datetime.now()
        similarity_mat = random_Walk_ECPCS(similarity_mat, self.l)
        end_Time = datetime.datetime.now()
        print('randwalk:'+str(end_Time-start_time))
        # index_return = index_return.astype(np.int64)
        rho = self.rho_creator(Z)
        # sim, nneigh = simNeigh_creator(rm_similiarity, rho, p, k)
        start_time = datetime.datetime.now()
        nneigh, sim, sort_rho_idx = self.simNeigh_creator(similarity_mat, rho)
        end_Time = datetime.datetime.now()
        print('NDH:'+str(end_Time-start_time))

        core_index = [x for (x, y) in enumerate(nneigh) if y == -1]
        start_time = datetime.datetime.now()

        nneigh, topK = self.DC_inter_dominance_estimation(rho, nneigh, core_index, sort_rho_idx, similarity_mat)
        end_Time = datetime.datetime.now()
        # print('dc:'+str(end_Time-start_time))
        if type(topK) == int:
            importance_sorted = self.topK_selection(sim, rho, self.k)
            self.sample_cluster(sort_rho_idx, nneigh)
        # print(self.center)
        # print(self.cluster)
        del similarity_mat, nneigh
        self.final_cluster()
        # index_return = self.index_return.astype(np.int32)
        # index_return = self.index_return.to('cpu').numpy().astype(np.int32)
        # self.sample_label = type(self.cluster)(map(lambda i:self.cluster[i], index_return))
        return self.sample_label, self.label, self.loadtime


    def sample_Data(self):
        tmp_minDistance, tmp_index = self.data_to_kms.min(axis=1)   # The minimum distance from each input sample to the Kmeans cluster center
        # tmp_index = self.data_to_kms.argmin(axis=1)      # Index of the minimum distance from each input sample to the Kmeans cluster center
        # data_to_kms = data_to_kms[:, shulffle_N[0:self.s]]
        sigma = torch.sqrt(torch.mean(torch.mean(self.data_to_kms)))

        minDistance = torch.zeros([self.p, self.r])
        index = torch.zeros([self.p, self.r])
        minDistance[:, 0] = tmp_minDistance
        index[:, 0] = tmp_index

        starttime = datetime.datetime.now()
        similarity_mat, Z = self.similarity_creator(self.data_to_kms, minDistance, index, sigma)
        endtime = datetime.datetime.now()
        print("LDP:"+str(endtime-starttime))
        similarity_mat = mapminmax(similarity_mat, 1, 0)
        return similarity_mat, Z

    def similarity_creator(self, Eudist_mat, minEudist_mat, minEudist_index, sigma):
        """
            Get the adjacency matrix and the similarity matrix
            params:
                Eudist_mat: distance matrix
                minEudist_mat: mininum distances of the points
                minEudist_index: index of the mininum distances
                sigma: threshold
            returns:
                similarity_mat: similarity matrix
                Z: adjacency matrix
        """
        j = 1
        starttime = datetime.datetime.now()
        for t in range(0, self.r - 1):
            Eudist_mat[((torch.Tensor(range(0, self.p))).reshape(-1,1)).tolist(), (minEudist_index[0: self.p, (j-1)]).reshape(-1,1).tolist()] = 1e10
            minEudist_mat[0:self.p, j], minEudist_index[0: self.p, j] = torch.min(Eudist_mat, dim=1)
            j = j + 1
        # print(minEudist_mat)
        endtime = datetime.datetime.now()
        print(endtime - starttime)
        minEudist_mat = torch.exp(-minEudist_mat / (2 * sigma**2))
        # print(torch.Tensor(range(0, self.p)))
        row = torch.unsqueeze(torch.transpose(torch.Tensor(range(0, self.p)), -1, 0), 1).repeat(1, self.r)
        col = minEudist_index.type(torch.IntTensor)
        index_input = torch.Tensor(list(zip(row.flatten(), col.flatten())))
        index_input = torch.transpose(index_input, -1, 0)
        index_input = index_input.to(device)
        # Z = coo_matrix((minEudist_mat.flatten(), (row.flatten(), col.flatten())), shape=(self.p, self.s)).toarray()    # Z: adjacency matrix
        Z = torch.sparse_coo_tensor(index_input, minEudist_mat.to(device).flatten(), torch.Size([self.p, self.s]))
        Z = torch.transpose(Z, 0, 1)
        starttime = datetime.datetime.now()
        Z = mapminmax(Z, 1, 0)
        endtime = datetime.datetime.now()
        print('z:'+str(endtime-starttime))
        similarity_mat = torch.mm(torch.transpose(Z,0,1), Z)     # similarity_mat: similarity matrix
        return similarity_mat, Z

    def rho_creator(self, Z):
        """
            Get the density of each sample from the adjacency matrix
            params:
                Z: adjacency matrix
            returns:
                rho: density matrix
        """
        rho = torch.sum(Z, axis=0)
        rho = mapminmax(rho, 1, 0)
        rho = rho.flatten()
        
        return rho

    def f(self, x):
        # non zero values mask
        non_zero_mask = x > 0

        # operations on the mask to find first nonzero values in the rows
        mask_max_values, mask_max_indices = torch.max(non_zero_mask, dim=1)

        # if the max-mask is zero, there is no nonzero value in the row
        mask_max_indices[mask_max_values == 0] = -1
        return mask_max_indices

    def simNeigh_creator(self, similarity, rho):
        """
            Get the ranking of similarity, the nearest neighbour and the ranking of density from the similarity matrix
            params:
                similarity: similarity matrix
                rho: density matrix
            returns:
                sim: the ranking of similarity
                nneigh: the nearest neighbour of every point
                sort_rho_idx: the ranking of density
        """
        similarity_compared, similarity_compared_idx = torch.sort(similarity, descending=True)
        # print(similarity_compared_idx)
        # rho_repeat = rho.repeat(1, self.p)
        # index_input = torch.cat((torch.Tensor((range(0, self.p))).reshape((self.p, 1)).repeat(1, self.p).flatten().reshape((1, self.p*self.p)).to(device), similarity_compared_idx.flatten().reshape((1, self.p*self.p))), 0)
        # print(torch.transpose(rho_repeat, 0, 1))
        similarity_compared_idx_flatten = similarity_compared_idx.flatten()
        ldp_sorted = rho[similarity_compared_idx_flatten[[i for i in range(0, self.p*self.p)]]].reshape((self.p, self.p))
        # print(ldp_sorted)
        # ldp_sorted = torch.sparse_coo_tensor(index_input, torch.transpose(rho_repeat, 0, 1).flatten(), torch.Size([self.p, self.p])).to_dense()
        # print(ldp_sorted)

        current_rho = torch.transpose(rho.repeat(self.p, 1), 0, 1)
        ndh_finder = ldp_sorted - current_rho
        ndh_finder[:, 0] = -1
        # idx = torch.arange(ndh_finder.shape[1], 0, -1).to(device)
        # tmp = ndh_finder*idx
        # print(tmp)
        # ndh_idx = torch.argmax(tmp, 1, keepdim=True)
        ndh_idx = self.f(ndh_finder)
        nneigh = similarity_compared_idx[[i for i in range(0, self.p)], [ndh_idx[j] for j in range(0, self.p)]]
        sim = similarity[[i for i in range(0, self.p)], [nneigh[j] for j in range(0, self.p)]]
        sort_rho_idx = torch.argsort(-rho)
        sim[sort_rho_idx[0]] = 1e10
        # print(sim)
        sim[sort_rho_idx[0]] = torch.min(sim)
        nneigh[sort_rho_idx[0]] = -1
        sim = mapminmax(sim, 1, 0)
        sim = sim.flatten()
        return nneigh, sim, sort_rho_idx


    def topK_selection(self, sim, rho, k):
        """
            Select top-K significant DCs
            Similar to DPC, do importance estimates
            params:
                sim: the ranking of similarity
                rho: density matrix
            returns:
                importance_sorted: the ranking of importances
        """
        importance = []
        for i in range(0, len(sim)):
            sim[torch.argmin(sim)] = 0.01
        dist = 1.0 / sim
        # print('dist:',dist)
        # plt.scatter(rho, dist)
        # plt.show()
        for s, d in zip(dist, rho):
            importance.append(d * s)
        importance = (torch.Tensor(importance)).to(device)
        # importance = importance.flatten()
        importance_sorted = torch.argsort(-importance)
        for i in range(0, k):
            self.cluster[importance_sorted[i]] = i
            self.center[i] = importance_sorted[i]
        return importance_sorted

    def sample_cluster(self, importance_sorted, nneigh):
        """
            Use the topK centers to cluster the modes
            params:
                importance_sorted: the ranking of importances
                nneigh: nearst neighbour of every point
        """
        for i in range(0, self.p):
            if self.cluster[importance_sorted[i]] == -1:
                self.cluster[importance_sorted[i]] = self.cluster[nneigh[importance_sorted[i]]]

    def DC_inter_dominance_estimation(self, rho, ndh, core_index, sort_rho_idx, similarity_mat):
        """
            From yanggeping's FastDEC
            params:
                rho: density matrix
                ndh: nearst density-higher
                core-index: someone has not ndh
                sort_rho_idx: ranking of density
                similarity_mat: similarity matrix
            return:
                topK_idx: the index of topK points
        """
        # core_index = np.array(core_index)
        sim = similarity_mat[core_index, :]
        indices = torch.argsort(-sim, axis=1)
        # sim = np.array(sim)
        sim = torch.sort(sim, axis=1).values
        sim = sim.to('cpu')
        sim = sim.numpy()
        # print(sim.type())
        sim = sim[:, ::-1]
        sim = torch.from_numpy(np.ascontiguousarray(sim))
        num_core = len(core_index)
        g = np.full(num_core, -1, np.float32)
        g = torch.Tensor(g)
        for q, i in enumerate(core_index):
            for p, j in enumerate(indices[q]):
                if rho[i] < rho[j]:
                    g[q] = sim[q][p]
                    ndh[i] = j
                    break
        topK_idx = 0
        if self.k < num_core:
            for i in range(num_core):
                if g[i] == -1:
                    g[i] = torch.max(g)
            g = normalize_by_minmax(g)
            core_density = rho[core_index]
            core_density = normalize_by_minmax(core_density)
            SD = core_density * g
            # i = np.argsort(-SD)[0:self.k]
            topK_idx = core_index[np.argsort(-SD)[0:self.k]]
            label = [-1]*self.N
            count = 0
            for i in topK_idx:
                label[i] = count
                count += 1
            for i in sort_rho_idx:
                if self.cluster[i] == -1:
                    self.cluster[i] = label[ndh[i]]
        return ndh, topK_idx

    # def final_cluster(self, cdh_ids, core_idx, density):
    #     """
    #         From yanggeping's FastDEC
    #         Use after 'DC_inter_dominance_estimation'
    #         params:
    #             density: density matrix
    #             cdh_ids: nearst density-higher
    #             core-indx: someone has not ndh
    #         return:
    #             label: the labels of clustering
    #     """
    #     label = torch.full(self.n, -1, np.int8)
    #     sorted_density = torch.argsort(-density)
    #     count = 0
    #     for i in core_idx:
    #         label[i] = count
    #         count += 1
    #     for i in sorted_density:
    #         if label[i] == -1:
    #             label[i] = label[cdh_ids[i]]
    #     return label
    def remaining_memory(self):
        """
        Get remaining memory in gpu
        """
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        if _pynvml_exist:
            pynvml.nvmlInit()
            gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
            remaining = info.free
        else:
            remaining = torch.cuda.memory_allocated()
        return remaining

    def final_cluster(self):
        fig = 0
        fig2 = 0
        for i in range(0, 16):

            load_starttime = datetime.datetime.now()
            torch.cuda.empty_cache()
            data, label_tmp = get_8m(file_Name[i])
            if self.label == []:
                self.label = label_tmp
            else:
                self.label = np.r_[self.label, label_tmp]
            N = data.shape[0]
            data = torch.Tensor(data).to(device)
            load_endtime = datetime.datetime.now()
            if fig2 == 0:
                self.loadtime = (load_endtime-load_starttime)
                fig2 = 1
            else:
                self.loadtime += (load_endtime-load_starttime)
            # data = dim_reduction(data, 500)  # dimension reduction by PCA
            # batch_size = data.shape[0]
            # if data.dtype == torch.double:
            #     expected = data.shape[0] * data.shape[1] * 8
            # if data.dtype == torch.float:
            #     expected = data.shape[0] * data.shape[1] * 4
            # ratio = math.ceil(expected / self.remaining_memory())
            # sub_batch_size = math.ceil(batch_size / ratio)
            # data_to_kms = torch.zeros((N, self.p))

            starttime = datetime.datetime.now()
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            
            min_idx = None
            kmeans = KMeans(n_clusters=self.p, mode='euclidean', verbose=1, max_iter=1)
            for k in range(0, 5):
                min_idx_tmp, __ = kmeans.fit_predict(data[(k)*100000:((k+1)*100000),:], centroids=self.kmeans_center)
                if min_idx == None:
                    min_idx = min_idx_tmp
                else:
                    min_idx = torch.cat((min_idx, min_idx_tmp))
            # for j in range(ratio):
            #     # data = data[(int)(j*sub_batch_size):((int)((j+1)*sub_batch_size)),:]
            #     data_to_kms_ = Euc_dist(data[(int)(j*sub_batch_size):((int)((j+1)*sub_batch_size)),:].to('cpu').numpy(), self.kmeans_center.to('cpu').numpy())  # The distance matrix from each input sample to the Kmeans cluster center
            #     data_to_kms[(int)(j*sub_batch_size):((int)((j+1)*sub_batch_size)), :] = data_to_kms_
            # endtime = datetime.datetime.now()
            # min_idx = data_to_kms.argmin(axis=1)
            # print(min_idx)
            self.sample_label[i*N:(i+1)*N] = self.cluster.repeat(N, 1)[np.stack((torch.Tensor([j for j in range(0,N)]), min_idx.to('cpu')),axis=0)]
            # self.sample_label[i*N:(i+1)*N] = type(self.cluster)(map(lambda k:self.cluster[k], min_idx))
                        
            endtime = datetime.datetime.now()
            if fig == 0:
                alltime = (endtime-starttime)
                fig = 1
            else:
                alltime += (endtime-starttime)
            del data, min_idx, label_tmp
        print("dist:"+str(alltime))
            # data_to_kms = data_to_kms[shulffle_N[0:self.s], :]
            # data_to_kms_alltmp = torch.transpose(data_to_kms, 0, 1)
            # tmp_index2_all = np.r_[tmp_index2_all, tmp_index2_alltmp.numpy()]
        # data_to_kms_all = np.c_[data_to_kms_all, data_to_kms_alltmp]

def get_prototype(data, s, p):
    N = data.shape[0]
    print(N)
    shulffle_N = np.array(range(0, N))
    np.random.seed(5)
    np.random.shuffle(shulffle_N)
    # The first S of the input data are taken for Kmeans clustering
    kmeans_data = data[shulffle_N[0:s], :]
    torch.cuda.empty_cache()
    # kmeans = KMeans(n_clusters=p, max_iter=3,random_state=0).fit(kmeans_data.to('cpu').numpy())
    
    # kms_label, kms_center = kmeans(X=kmeans_data, num_clusters=self.p, device=device)
    starttime = datetime.datetime.now()
    kmeans = KMeans(n_clusters=p, mode='euclidean', verbose=1)
    endtime = datetime.datetime.now()
    print("kmeans:"+str(endtime-starttime))

    kms_label, kms_center = kmeans.fit_predict(kmeans_data)

    # kms_center = kmeans.cluster_centers_    # Centers after kmeans clustering
    kms_center = kms_center.to(device)
    data_to_kms = torch.zeros((N, p))
    for b in range(0, 5):
        data_tmp = data[(int)(N*b/5):((int)(N*(b+1)/5)),:]
        data_to_kms_ = Euc_dist(data_tmp.to('cpu').numpy(), kms_center.to('cpu').numpy())  # The distance matrix from each input sample to the Kmeans cluster center
        data_to_kms[(int)(N*b/5):((int)(N*(b+1)/5)), :] = data_to_kms_
    # tmp_index2_all = data_to_kms.argmin(axis=1)
    data_to_kms = data_to_kms[shulffle_N[0:s], :]
    data_to_kms = torch.transpose(data_to_kms, 0, 1)
    return kms_center, data_to_kms

