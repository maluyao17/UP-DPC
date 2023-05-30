from sklearn import metrics
from Dataprocessing import get_emnist, load_Data, load_Data2, get_data, get_data2, get_8m, get_data3
# from Dpcpp_ceneter_Ldp import Dpcpp
from UP_Auto import Dpcpp
# from Dpcpp_compare import Dpcpp
# from Dpcpp_test import Dpcpp
# from Dpcpp_density import Dpcpp
# from Dpcpp_batch import Dpcpp
import numpy as np
import torch
# ARI_avg = 0.0
# AMI_avg = 0.0
# NMI_avg = 0.0
file_Name = '../Dpcpp/datasets/mnist.mat'
# file_Name = '../Dpcpp/datasets/mnist_2d.csv'
# file_Name = 'E:/mnist8m/mnist1q.mat'
l = 50
print(file_Name)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data, org_label = get_data(file_Name)
# data, label = get_data(file_Name)
# data, label = get_emnist(file_Name)
print(torch.cuda.is_available())
data = torch.Tensor(data).to(device)
opreator = Dpcpp(data, org_label, l)
sample_label, time = opreator.fit()
print(sample_label)
sample_label = np.array(torch.tensor(sample_label, device='cpu'), dtype=np.int8)
# org_label = label.to('cpu')
ARI = metrics.adjusted_rand_score(org_label, sample_label)
AMI = metrics.adjusted_mutual_info_score(org_label, sample_label)
NMI = metrics.normalized_mutual_info_score(org_label, sample_label)

print('s=', opreator.s, ' p=', opreator.p, ' l=', l)
print('total time=', time)
print("Adj. Rand Index Score=", ARI)
print("Adj. Mutual Info Score=", AMI)
print("Norm Mutual Info Score=", NMI)
str__ = "\nfile_name:" + str(file_Name) + "\n + "+ "s:"+ str(opreator.s) + "p:" + str(opreator.p) + ",l:" + str(l)  + ", time: " + str(time) + "\nNMI:"+str(NMI)+',AMI:'+str(AMI)+',ARI:'+str(ARI)
with open("./result/auto_batch.txt", "a+") as f:
    f.write(str__)