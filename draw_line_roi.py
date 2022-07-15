
import numpy  as np
import pylab as plt
from PyEMD import EMD
def draw(y1,path="",xuhao=""):
    x = np.arange(len(y1))
    plt.plot(x,y1)
    plt.title(path+xuhao)
    plt.show()
pathp = "D:/dataset/micro_datatset/test_SAMM_crop5/006_1/1136.npy"
print(pathp)
a=np.load(pathp)
print(a.shape)
flow200=np.load(pathp)
width=int(flow200.shape[0]/18)

flow200 = np.array(flow200)
flow200 = np.sqrt(np.sum(flow200 ** 2, axis=1))

for i in range(18):
    f=flow200[width*(i): width*(i+1)-1]
    draw(f, path=pathp, xuhao=str(i))

