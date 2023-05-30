from operator import index
from sklearn import metrics
from Dataprocessing import get_8m, load_Data, load_Data2, get_data, get_data2
from UP_largescale_batch_Auto import Dpcpp
import numpy as np
import scipy.io as sio
import datetime
import torch
import UP_largescale_batch_Auto


file_Name = ['../Dpcpp/datasets/mnist1q.mat','../Dpcpp/datasets/mnist2q.mat',
                '../Dpcpp/datasets/mnist3q.mat','../Dpcpp/datasets/mnist4q.mat',
                '../Dpcpp/datasets/mnist5q.mat','../Dpcpp/datasets/mnist6q.mat',
                '../Dpcpp/datasets/mnist7q.mat','../Dpcpp/datasets/mnist8q.mat',
                '../Dpcpp/datasets/mnist9q.mat','../Dpcpp/datasets/mnist10q.mat',
                '../Dpcpp/datasets/mnist11q.mat','../Dpcpp/datasets/mnist12q.mat',
                '../Dpcpp/datasets/mnist13q.mat','../Dpcpp/datasets/mnist14q.mat',
                '../Dpcpp/datasets/mnist15q.mat','../Dpcpp/datasets/mnist16q.mat']


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

l = 50
# s=[700]
# p=[30]
# l=[10]
print(torch.cuda.is_available())
starttime = datetime.datetime.now()
# label = []
data = []
index_return = []
data_to_kms = []
print(file_Name)
data_tmp, label_tmp = get_8m(file_Name[0])

data_tmp = torch.Tensor(data_tmp).to(device)

kms_center, data_to_kms, N, s, p = UP_largescale_batch_Auto.get_prototype(data_tmp)

opreator = Dpcpp(label_tmp, kms_center, data_to_kms, s, p, l, fn=16, N=data_tmp.shape[0])
sample_label, org_label, loadtime = opreator.fit()
endtime = datetime.datetime.now()
time = endtime - starttime - loadtime
sample_label = np.array(torch.tensor(sample_label, device='cpu'), dtype=np.int8)
org_label = np.array(org_label, dtype=np.int8)
print(sample_label)
ARI = metrics.adjusted_rand_score(org_label, sample_label)
AMI = metrics.adjusted_mutual_info_score(org_label, sample_label)
NMI = metrics.normalized_mutual_info_score(org_label, sample_label)
print('s=', s, ' p=', p, ' l=', l)
print('total time=', time)
print("Adj. Rand Index Score=", ARI)
print("Adj. Mutual Info Score=", AMI)
print("Norm Mutual Info Score=", NMI)

str__ = "\nfile_name:" + str(file_Name) + "\ns:" + str(s) + ",p:" + str(p) + ",l:" + str(l)  + ", time: " + str(time) + "\nNMI:"+str(NMI)+',AMI:'+str(AMI)+',ARI:'+str(ARI)
with open("dpc_8m_batch.txt", "a+") as f:
    f.write(str__)
