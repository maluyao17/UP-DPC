import random
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.preprocessing import MinMaxScaler
from Dataprocessing import mapminmax
from Dataprocessing import load_Data, load_Data2, get_data, Euc_dist, normalize_by_minmax
import datetime
import random
import matplotlib.pylab as plt
import torch
from fast_pytorch_kmeans import KMeans
import seaborn as sns
import math
sns.set()
try:
    import pynvml
    _pynvml_exist = True
except ModuleNotFoundError:
    _pynvml_exist = False

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
    return newS


class Dpcpp:

    def __init__(self, data, label, s, p, l, r=100):
        self.data = data
        self. label = label
        self.s = s
        self.p = p
        self.l = l
        self.r = r
        self.k = len(set((label.tolist())))
        self.center = [-1] * self.k

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


    def fit(self):
        starttime_all = datetime.datetime.now()
        similarity_mat, Z, N, index_return, kmeans_label = self.sample_Data()
        walk_start = datetime.datetime.now()
        endtime = datetime.datetime.now()
        similarity_mat = random_Walk_ECPCS(similarity_mat, self.l)
        walk_end = datetime.datetime.now()
        print('Rand Walk:'+ str(walk_end-walk_start))
        endtime = datetime.datetime.now()

        rho = self.rho_creator(Z)
        index_return = index_return.int()
        starttime = datetime.datetime.now()
        nneigh, sim, sort_rho_idx = self.simNeigh_creator(similarity_mat, rho)
        endtime = datetime.datetime.now()
        print("NDH:"+str(endtime - starttime))
        core_index = [x for (x, y) in enumerate(nneigh) if y == -1]
        nneigh, topK = self.DC_inter_dominance_estimation(rho, nneigh, core_index, sort_rho_idx, similarity_mat)
        self.plot_heatmap(similarity_mat)
        if type(topK) == int:
            importance_sorted = self.topK_selection(sim, rho, self.k)
            self.sample_cluster(sort_rho_idx, nneigh)
        self.data = self.data.to('cpu')
        self.shulffle = self.shulffle.to('cpu')
        self.prototype = self.prototype.to('cpu')
        self.top_k = self.prototype[importance_sorted[0:self.k],:]
        # self.center = self.center.to('cpu')
        self.plot_sample1(self.data,self.shulffle,self.prototype,self.top_k)
        self.plot_sample2(self.data,self.shulffle,self.prototype,self.top_k)
        self.plot_sample3(self.data,self.shulffle,self.prototype,self.top_k)
        self.plot_sample4(self.data,self.shulffle,self.prototype,self.top_k)
        self.plot0 = []
        self.plot1 = []
        self.plot2 = []
        self.plot3 = []
        self.plot4 = []
        self.plot5 = []
        self.plot6 = []
        self.plot7 = []
        self.plot8 = []
        self.plot9 = []


        # for i in range(0, N):
        #     self.sample_label[i] = self.cluster[index_return[i]]
        #     if self.cluster[index_return[i]] == 0:
        #         self.plot0.append(i)
        #     elif self.cluster[index_return[i]] == 1:
        #         self.plot1.append(i)
        #     elif self.cluster[index_return[i]] == 2:
        #         self.plot2.append(i)
        #     elif self.cluster[index_return[i]] == 3:
        #         self.plot3.append(i)
        #     elif self.cluster[index_return[i]] == 4:
        #         self.plot4.append(i)
        #     elif self.cluster[index_return[i]] == 5:
        #         self.plot5.append(i)
        #     elif self.cluster[index_return[i]]== 6:
        #         self.plot6.append(i)
        #     elif self.cluster[index_return[i]] == 7:
        #         self.plot7.append(i)
        #     elif self.cluster[index_return[i]] == 8:
        #         self.plot8.append(i)
        #     elif self.cluster[index_return[i]] == 9:
        #         self.plot9.append(i)
        self.plot_M2D(self.data,self.sample_label)
        index_return = index_return.to('cpu')
        del self.data, similarity_mat, self.prototype, self.shulffle
        torch.cuda.empty_cache()
        self.sample_label = self.cluster.repeat(N, 1)[np.stack((torch.Tensor([i for i in range(0,N)]), index_return),axis=0)]
        endtime = datetime.datetime.now()
        return self.sample_label, (endtime - starttime_all)

    def sample_Data(self):
        N = self.data.shape[0]
        self.cluster = torch.Tensor([-1] * self.p).to(device)
        self.N = N
        self.sample_label = [-1] * N
        print(N)
        shulffle_N = np.array(range(0, N))
        np.random.seed(5)
        np.random.shuffle(shulffle_N)
        kmeans_data = self.data[shulffle_N[0:self.s], :]
        print('memory:'+str(self.remaining_memory()))
        self.shulffle = kmeans_data
        starttime = datetime.datetime.now()
        torch.cuda.empty_cache()
        kmeans = KMeans(n_clusters=self.p, mode='euclidean', verbose=1)
        kms_label, kms_center = kmeans.fit_predict(kmeans_data)
        endtime = datetime.datetime.now()
        print('kmeans algorithm:'+ str(endtime - starttime))
        kms_center = kms_center.to(device)
        self.prototype = kms_center
        batch_size = kmeans_data.shape[0]
        if kmeans_data.dtype == torch.double:
            expected = kmeans_data.shape[0] * kmeans_data.shape[1] * 8
        if kmeans_data.dtype == torch.float:
            expected = kmeans_data.shape[0] * kmeans_data.shape[1] * 4
        ratio = math.ceil(expected / self.remaining_memory())
        sub_batch_size = math.ceil(batch_size / ratio)
        data_to_kms = torch.zeros((self.s, self.p))
        print('memory:'+str(self.remaining_memory()))


        for i in range(ratio):
            data = kmeans_data[(int)(i*sub_batch_size):((int)((i+1)*sub_batch_size)),:]
            data_to_kms_ = Euc_dist(data.to('cpu').numpy(), kms_center.to('cpu').numpy())  # The distance matrix from each input sample to the Kmeans cluster center
            data_to_kms[(int)(i*sub_batch_size):((int)((i+1)*sub_batch_size)), :] = data_to_kms_
            print('memory:'+str(self.remaining_memory()))

        print('memory:'+str(self.remaining_memory()))

        starttime = datetime.datetime.now()
        kmeans = KMeans(n_clusters=self.p, mode='euclidean', verbose=1, max_iter=1)
        tmp_index2,__ = kmeans.fit_predict(self.data, centroids=kms_center)
        endtime = datetime.datetime.now()
        print('Calculate the kmeans distance:'+ str(endtime - starttime))
        data_to_kms = torch.transpose(data_to_kms, 0, 1)
        tmp_minDistance, tmp_index = data_to_kms.min(axis=1)   # The minimum distance from each input sample to the Kmeans cluster center

        sigma = torch.sqrt(torch.mean(torch.mean(data_to_kms)))

        minDistance = torch.zeros([self.p, self.r])
        index = torch.zeros([self.p, self.r])
        minDistance[:, 0] = tmp_minDistance
        index[:, 0] = tmp_index
        print(datetime.datetime.now()-endtime)
        starttime = datetime.datetime.now()
        similarity_mat, Z = self.similarity_creator(data_to_kms.to(device), minDistance.to(device), index.to(device), sigma.to(device))
        endtime = datetime.datetime.now()
        print('LDP graph:'+str(endtime-starttime))

        similarity_mat = mapminmax(similarity_mat, 1, 0)

        return similarity_mat, Z, N, tmp_index2, kms_label

    def similarity_creator(self, Eudist_numpy, minEudist_mat, minEudist_index, sigma):
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
            Eudist_numpy[((torch.Tensor(range(0, self.p))).reshape(-1,1)).tolist(), (minEudist_index[0: self.p, (j-1)]).reshape(-1,1).tolist()] = 1e10
            minEudist_mat[0:self.p, j], minEudist_index[0: self.p, j] = torch.min(Eudist_numpy, dim=1)
            j = j + 1
        # print(minEudist_mat)

        minEudist_mat = torch.exp(-minEudist_mat / (2 * sigma**2))
        # print(torch.Tensor(range(0, self.p)))
        row = torch.unsqueeze(torch.transpose(torch.Tensor(range(0, self.p)), -1, 0), 1).repeat(1, self.r)
        col = minEudist_index.type(torch.IntTensor)
        index_input = torch.Tensor(list(zip(row.flatten(), col.flatten())))
        index_input = torch.transpose(index_input, -1, 0)
        index_input = index_input.to(device)
        Z = torch.sparse_coo_tensor(index_input, minEudist_mat.flatten(), torch.Size([self.p, self.s]))
        Z = torch.transpose(Z, 0, 1)
        Z = mapminmax(Z, 1, 0)
        similarity_mat = torch.mm(torch.transpose(Z,0,1), Z)     # similarity_mat: similarity matrix
        endtime = datetime.datetime.now()
        print(endtime - starttime)
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
        # rho = mapminmax(rho, 1, 0)
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
        similarity_compared, similarity_compared_idx = torch.sort(similarity, descending=True)
        similarity_compared_idx_flatten = similarity_compared_idx.flatten()
        ldp_sorted = rho[similarity_compared_idx_flatten[[i for i in range(0, self.p*self.p)]]].reshape((self.p, self.p))

        current_rho = torch.transpose(rho.repeat(self.p, 1), 0, 1)
        ndh_finder = ldp_sorted - current_rho
        ndh_finder[:, 0] = -1
        ndh_idx = self.f(ndh_finder)
        nneigh = similarity_compared_idx[[i for i in range(0, self.p)], [ndh_idx[j] for j in range(0, self.p)]]
        sim = similarity_compared[[i for i in range(0, self.p)], [ndh_idx[j] for j in range(0, self.p)]]
        sort_rho_idx = torch.argsort(-rho)
        sim[sort_rho_idx[0]] = 1e10
        sim[sort_rho_idx[0]] = torch.min(sim)
        nneigh[sort_rho_idx[0]] = -1
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
        importance = importance.flatten()
        importance_sorted = torch.argsort(-importance)
        for i in range(0, k):
            self.cluster[importance_sorted[i]] = i
            self.center[i] = importance_sorted[i]
        return importance_sorted

    def sample_cluster(self, rho_sorted, nneigh):
        """
            Use the topK centers to cluster the modes
            params:
                importance_sorted: the ranking of importances
                nneigh: nearst neighbour of every point
        """
        for i in range(0, self.p):
            if self.cluster[rho_sorted[i]] == -1:
                # print(nneigh[rho_sorted[i]])
                self.cluster[rho_sorted[i]] = self.cluster[nneigh[rho_sorted[i]]]

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

    def final_cluster(self, cdh_ids, core_idx, density):
        """
            From yanggeping's FastDEC
            Use after 'DC_inter_dominance_estimation'
            params:
                density: density matrix
                cdh_ids: nearst density-higher
                core-indx: someone has not ndh
            return:
                label: the labels of clustering
        """
        label = np.full(self.n, -1, np.int8)
        sorted_density = np.argsort(-density)
        count = 0
        for i in core_idx:
            label[i] = count
            count += 1
        for i in sorted_density:
            if label[i] == -1:
                label[i] = label[cdh_ids[i]]
        return label

    def plot_sample1(self, input, shulffle, prototype, top_k):
        plt.style.use('default')
        plt.figure(figsize=[6.40, 5.60])
        plt.scatter(x=input[:,0], y=input[:,1], marker=',',c='gray',alpha=0.5)
        # plt.title('MNIST_2D dataset')
        plt.show()

    def plot_sample2(self, input, shulffle, prototype, top_k):
        plt.style.use('default')
        plt.figure(figsize=[6.40, 5.60])
        plt.scatter(x=input[:,0], y=input[:,1], marker=',',c='gray',alpha=0.5)
        print(input[:,0])
        plt.scatter(x=shulffle[:,0], y=shulffle[:,1], marker='.',c='black',alpha=0.75)
        # plt.title("Sampled MNIST_2D dataset")
        plt.show()

    def plot_sample3(self, input, shulffle, prototype, top_k):
        plt.style.use('default')
        plt.figure(figsize=[6.40, 5.60])
        plt.scatter(x=input[:,0], y=input[:,1], marker=',',c='gray',alpha=0.5)
        plt.scatter(x=shulffle[:,0], y=shulffle[:,1], marker='.',c='black',alpha=0.75)
        plt.scatter(x=prototype[:,0], y=prototype[:,1], marker='o',c='blue',alpha=0.95)
        # plt.title("Prototypes in the MNIST_2D dataset")
        plt.show()

    def plot_sample4(self, input, shulffle, prototype, top_k):
        plt.style.use('default')
        plt.figure(figsize=[6.40, 5.60])
        plt.scatter(x=input[:,0], y=input[:,1], marker=',',c='gray',alpha=0.5)
        plt.scatter(x=shulffle[:,0], y=shulffle[:,1], marker='.',c='black',alpha=0.75)
        plt.scatter(x=prototype[:,0], y=prototype[:,1], marker='o',c='blue',alpha=0.95)
        plt.scatter(x=top_k[:,0], y=top_k[:,1], marker='*',c='red', s=200)
        # plt.title("Modes in the MNIST_2D dataset")
        plt.show()

    def plot_M2D(self, data, label):
        plt.figure(figsize=[6.40, 5.60])
        # plt.rcParams['axes.facecolor']='white'
        plt.style.use('default')
        # plt.grid(False)
        # print((data[self.plot0, 0], data[self.plot0, 1]))
        plt.scatter(data[self.plot0, 0], data[self.plot0, 1], marker='.', c='#FFA500', alpha=0.5)
        plt.scatter(data[self.plot1, 0], data[self.plot1, 1], marker='.', c='#87CEEB', alpha=0.5)
        plt.scatter(data[self.plot2, 0], data[self.plot2, 1], marker='.', c='#FFB6C1', alpha=0.5)
        plt.scatter(data[self.plot3, 0], data[self.plot3, 1], marker='.', c='#D3D3D3', alpha=0.5)
        plt.scatter(data[self.plot4, 0], data[self.plot4, 1], marker='.', c='#D8BFD8', alpha=0.5)
        plt.scatter(data[self.plot5, 0], data[self.plot5, 1], marker='.', c='#FA8072', alpha=0.5)
        plt.scatter(data[self.plot6, 0], data[self.plot6, 1], marker='.', c='#F08080', alpha=0.5)
        plt.scatter(data[self.plot7, 0], data[self.plot7, 1], marker='.', c='#4B0082', alpha=0.5)
        plt.scatter(data[self.plot8, 0], data[self.plot8, 1], marker='.', c='#DEB887', alpha=0.5)
        plt.scatter(data[self.plot9, 0], data[self.plot9, 1], marker='.', c='#808000', alpha=0.5)
        plt.scatter(x=self.top_k[:,0], y=self.top_k[:,1], marker='*',c='red', s=200)

        # plt.title('mnist result')

        plt.show()

    def plot_s2(self, data, label):
        plt.style.use('default')
        plt.figure(figsize=[6.40, 5.60])
        for i in range(0, len(label)):
            if label[i] == 0:
                plt.scatter(data[i, 0], data[i, 1], marker='*', c='#FFA500', alpha=0.5)
            elif label[i] == 1:
                plt.scatter(data[i, 0], data[i, 1], marker='*', c='#87CEEB', alpha=0.5)
            elif label[i] == 2:
                plt.scatter(data[i, 0], data[i, 1], marker='*', c='#FFB6C1', alpha=0.5)
            elif label[i] == 3:
                plt.scatter(data[i, 0], data[i, 1], marker='*', c='#D3D3D3', alpha=0.5)
            elif label[i] == 4:
                plt.scatter(data[i, 0], data[i, 1], marker='.', c='#D8BFD8', alpha=0.5)
            elif label[i] == 5:        
                plt.scatter(data[i, 0], data[i, 1], marker='.', c='#FA8072', alpha=0.5)
            elif label[i] == 6:          
                plt.scatter(data[i, 0], data[i, 1], marker='.', c='#F08080', alpha=0.5)
            elif label[i] == 7:        
                plt.scatter(data[i, 0], data[i, 1], marker='.', c='#4B0082', alpha=0.5)
            elif label[i] == 8:
                plt.scatter(data[i, 0], data[i, 1], marker='.', c='#DEB887', alpha=0.5)
            elif label[i] == 9:
                plt.scatter(data[i, 0], data[i, 1], marker='.', c='#808000', alpha=0.5)
            elif label[i] == 10:
                 plt.scatter(data[i, 0], data[i, 1], marker='*', c='#FFD700', alpha=0.5)
            elif label[i] == 11:
                plt.scatter(data[i, 0], data[i, 1], marker='*', c='#ADFF2F', alpha=0.5)
            elif label[i] == 12:
                plt.scatter(data[i, 0], data[i, 1], marker='*', c='#2E8B57', alpha=0.5)
            elif label[i] == 13:
                plt.scatter(data[i, 0], data[i, 1], marker='*', c='#FFDAB9', alpha=0.5)
            elif label[i] == 14:
                plt.scatter(data[i, 0], data[i, 1], marker='*', c='#FF8C00', alpha=0.5)

        plt.show()

    def plot_desicion(self, topK, rho, sim):
        plt.figure(figsize=[6.40, 5.60])
        plt.style.use('default')
        rho = rho.to('cpu')
        sim = sim.to('cpu')
        for i in range(0, len(rho)):
            if i in topK[0:10]:
                plt.scatter(x=rho[i], y=1/sim[i], marker='*',c='red',alpha=0.95, s=200)
            else:
                plt.scatter(x=rho[i], y=1/sim[i], marker='.',c='blue',alpha=0.95)
        plt.xlabel(chr(961))
        plt.ylabel(chr(948))
        # plt.title('Desicion Map')
        plt.show()

    def plot_heatmap(self, mat):
        plt.figure(figsize=[6.40, 5.60])
        mat = mat.to('cpu')
        mask = np.zeros_like(mat)
        sns_plot = sns.heatmap(mat, mask=mask, xticklabels=100, yticklabels=100)


        sns_plot.tick_params(labelsize=15)
        cax = plt.gcf().axes[-1]
        cax.tick_params(labelsize=15)
        plt.show()