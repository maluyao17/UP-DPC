import csv
import numpy as np
from sklearn import preprocessing
import scipy.io as sio
from sklearn.metrics.pairwise import pairwise_distances
# from torchmetrics.functional import pairwise_euclidean_distance
import random
# import torch
import h5py
import datetime
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def load_Data(filename):
    """
        load_Data: load '.csv' files
        returns: data of array inputs& original label of array inputs
    """
    data = []
    label = []
    with open(filename, 'r') as file_obj:
        file = csv.reader(file_obj)
        for row in file:
            per_data = []
            for i in row[:-1]:
                per_data.append(float(i))
            data.append(per_data)
            label.append(float(row[-1]))
    data_arr = np.array(data)
    label_arr = np.array(label, np.int8)
    # data_arr = normalization(data_arr)
    mapminmax = preprocessing.MinMaxScaler()
    data_arr = mapminmax.fit_transform(data_arr)
    return data_arr, label_arr

def load_Data2(filename):
    """
        load_Data2: load '.csv' files
        returns: data of array inputs
    """
    data = []
    with open(filename, 'r') as file_obj:
        file = csv.reader(file_obj)
        for row in file:
            per_data = []
            for i in row[:]:
                per_data.append(float(i))
            data.append(per_data)
    data_arr = np.array(data)
    # data_arr = normalization(data_arr)
    # mapminmax = preprocessing.MinMaxScaler()
    # data_arr = mapminmax.fit_transform(data_arr)
    return data_arr

def get_data3(filename):
    """
        from yanggeping
        get_data: load '.mat' files
        returns:data of array inputs& original label of array inputs
    """
    # l= h5py.File(filename,'r','-v6')
    l = sio.loadmat(filename)
    data = l['fea']
    # print(type(data))
    # data = data.todense()
    # data = data.todense()
    # data = np.array(data)
    # print(data)
    label = l['gt']
    min_max_scaler = preprocessing.MinMaxScaler()
    # X_minMax = min_max_scaler.fit_transform(data)
    # X_minMax = data
    label = np.array(label).flatten()
    # print(label)
    return np.array(data, np.double), np.array(label, np.int32)

def get_data(filename):
    """
        from yanggeping
        get_data: load '.mat' files
        returns:data of array inputs& original label of array inputs
    """
    # l= h5py.File(filename,'r','-v6')
    l = sio.loadmat(filename)
    data = l['data']
    # print(type(data))
    # data = data.todense()
    # data = data.todense()
    # data = np.array(data)
    # print(data)
    label = l['label']
    min_max_scaler = preprocessing.MinMaxScaler()
    # X_minMax = min_max_scaler.fit_transform(data)
    # X_minMax = data
    label = np.array(label).flatten()
    # print(label)
    return np.array(data, np.float32), np.array(label, np.int32)

def get_data2(filename):
    """
        from yanggeping
        get_data: load '.mat' files
        returns:data of array inputs& original label of array inputs
    """
    l= h5py.File(filename,'r')
    # l = sio.loadmat(filename)
    data = l['fea']
    # print(type(data))
    # data = data.todense()
    # data = data.todense()
    data = np.array(np.transpose(data))
    # print(data)
    label = l['gnd']
    min_max_scaler = preprocessing.MinMaxScaler()
    # X_minMax = min_max_scaler.fit_transform(data)
    # X_minMax = data
    label = np.array(label).flatten()
    # print(label)
    return np.array(data, np.double), label

def get_emnist(filename):
    l = sio.loadmat(filename)
    dataset = l['dataset']
    train =dataset['train'][0, 0]
    test = dataset['test'][0, 0]

    train_data = train['images'][0, 0]
    train_label = train['labels'][0, 0]

    test_data = test['images'][0, 0]
    test_label = test['labels'][0, 0]

    data = np.vstack([train_data, test_data])
    label = np.vstack([train_label, test_label])
    min_max_scaler = preprocessing.MinMaxScaler()
    # X_minMax = min_max_scaler.fit_transform(data)
    # X_minMax = data
    label = np.array(label, np.float32).flatten()
    # print(label)
    return np.ascontiguousarray(np.array(data, np.float32)), np.array(label, np.int32)

def get_8m(filename):
    # l= h5py.File(filename,'r')
    l = sio.loadmat(filename)
    data = l['fea']
    # print(type(data))
    data = data.todense()
    data = np.array(data, np.float16)
    # data = data.todense()
    # data = np.array(np.transpose(data))
    # print(data)
    label = l['gnd']
    label = np.transpose(label)
    min_max_scaler = preprocessing.MinMaxScaler()
    # X_minMax = min_max_scaler.fit_transform(data)
    # X_minMax = data
    label = np.array(label).flatten()
    # print(label)
    return data ,np.array(label, np.int32)

def Euc_dist(mat_a, mat_b):
    """
        Euc_dist: Calculate the Euclidean distance between mat_a and mat_b
    """
    # mat_a = mat_a.unsqueeze(1)
    # mat_b = mat_b.unsqueeze(0)
    # distance_mat = pairwise_euclidean_distance(mat_a, mat_b)
    distance_mat = pairwise_distances(mat_a, mat_b)
    distance_mat = torch.Tensor(distance_mat).to(device)
    # distance_mat = np.power(distance_mat,2)
    return distance_mat

def mapminmax(arr, ymax, ymin):
    """
        Normalization of data
        It comes from function 'Mapminmax()' in MATLAB
        params:
            arr: array inputs
            ymax: Upper bounds of the outputs
            ymin: lower bounds of the outputs
        returns:
            arr_return: array outputs
    """
    # arr = np.array(arr)
    if len(arr.shape) == 1:
        arr = torch.reshape(arr, (arr.shape[0], 1))
        arr = torch.transpose(arr, 0, 1)
    print(arr.layout)
    if(arr.layout == torch.sparse_coo):
        arr = arr.to_dense()
    arr_b = torch.transpose((torch.min(arr, axis=1)).values, -1, 0)
    arr_b = torch.reshape(arr_b, (arr.shape[0], 1))
    arr_c = arr_b.repeat(1, arr.shape[1])
    arr_d = arr - arr_c

    arr_e = torch.transpose((torch.max(arr, axis=1)).values, -1, 0)
    arr_e = torch.reshape(arr_e, (arr.shape[0], 1))
    arr_f = arr_e.repeat(1, arr.shape[1])
    arr_g = arr_f - arr_c

    # ymax = torch.Tensor(ymax)
    # ymin = torch.Tensor(ymin)
    arr_return = torch.mul((ymax-ymin), arr_d)/(arr_g)+ymin
    find_nan = torch.nonzero(torch.isnan(arr_return))
    print(len(find_nan))
    arr_return[find_nan[:, 0],find_nan[:, 1]] = arr[find_nan[:, 0],find_nan[:, 1]]

    return arr_return

def normalize_by_minmax(data_sequence):
    """
        Normalization of data
        From yanggeping
    """
    min_v = np.min(data_sequence)
    max_v = np.max(data_sequence)
    range_v = max_v - min_v

    data_sequence = (data_sequence - min_v)/range_v

    return data_sequence

# def get_8m(filename):
#     # l= h5py.File(filename,'r')
#     l = sio.loadmat(filename)
#     data = l['fea']
#     # print(type(data))
#     data = data.todense()
#     # data = np.array(data)
#     # data = data.todense()
#     # data = np.array(np.transpose(data))
#     # print(data)
#     label = l['gnd']
#     label = np.transpose(label)
#     min_max_scaler = preprocessing.MinMaxScaler()
#     # X_minMax = min_max_scaler.fit_transform(data)
#     # X_minMax = data
#     label = np.array(label).flatten()
#     # print(label)
#     # return torch.from_numpy(data).to(device) ,torch.from_numpy(np.array(label, np.int32)).to(device)
#     return np.array(data*255, np.int8), np.array(label, np.int8)