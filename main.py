from sklearn import metrics
from Dataprocessing import get_emnist, load_Data, load_Data2, get_data, get_data2, get_8m, get_data3
from UP_ import Dpcpp
import numpy as np
import torch

file_Name = 'dataset/mnist_2D.csv'
# file_Name = '../Dpcpp/datasets/mnist_2d.csv'
# file_Name = 'E:/mnist8m/mnist1q.mat'
s = 6000
p = 500
l = 10
print(file_Name)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data, label = load_Data(file_Name)
print(torch.cuda.is_available())
data = torch.Tensor(data).to(device)
opreator = Dpcpp(data, label, s, p, l)
sample_label, time = opreator.fit()
sample_label = np.array(sample_label.to('cpu'), dtype=np.float64)
org_label = label
ARI = metrics.adjusted_rand_score(org_label, sample_label)
AMI = metrics.adjusted_mutual_info_score(org_label, sample_label)
NMI = metrics.normalized_mutual_info_score(org_label, sample_label)

print('s=', opreator.s, ' p=', opreator.p, ' l=', l)
print('total time=', time)
print("Adj. Rand Index Score=", ARI)
print("Adj. Mutual Info Score=", AMI)
print("Norm Mutual Info Score=", NMI)
str__ = "\nfile_name:" + str(file_Name) + "\n + "+ "s:"+ str(opreator.s) + "p:" + str(opreator.p) + ",l:" + str(l)  + ", time: " + str(time) + "\nNMI:"+str(NMI)+',AMI:'+str(AMI)+',ARI:'+str(ARI)
with open("./mnist_2d.txt", "a+") as f:
    f.write(str__)