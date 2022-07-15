import numpy as np
import matplotlib.pyplot as plt
import math
import cv2
import numpy as np
import scipy.signal as signal
import scipy.fftpack as fftpack
from sklearn.model_selection import KFold
import os


# 准备数据


# 过滤器
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    omega = 0.5 * fs  # fs是帧率
    low = lowcut / omega  #
    high = highcut / omega
    b, a = signal.butter(order, [low, high], btype='band')
    print(a.shape)
    print(b.shape)
    print(data.shape)
    y = signal.lfilter(b, a, data, axis=0)
    return y


def temporal_ideal_filter(tensor, low, high, fps, axis=0):
    fft = fftpack.fft(tensor, axis=axis)
    # print(fft)

    # x = np.arange(len(fft))
    # plt.plot(x, fft)
    # plt.show()
    frequencies = fftpack.fftfreq(tensor.shape[0], d=1.0 / fps)

    # bound_low = (np.abs(frequencies - low)).argmin()
    bound_high = (np.abs(frequencies - high)).argmin()
    # print(bound_high)
    # fft[1:-1] = 0
    # fft[1:bound_low] = 0
    # fft[0] = 0
    # fft[-2:-1] = 0
    fft[bound_high:-bound_high] = 0
    # fft[-bound_low:-1] = 0

    iff = fftpack.ifft(fft, axis=axis)

    return np.abs(iff)


def filt(y1, low, high, fps):
    # y1 = butter_bandpass_filter(y1, low, high, fps)
    y1 = temporal_ideal_filter(y1, low, high, fps)

    return y1


def draw(y1, path="", xuhao=""):
    x = np.arange(len(y1))
    plt.plot(x, y1)
    plt.title(path + xuhao)
    plt.show()


# nu = np.load("D:/aa_micro_expression/boshi/CAS(ME)2/test/dataset"+str(1)+".npy")
# nu=np.array(nu)
# nu=nu.reshape([-1,14,200])
# x=np.arange(200)
# for i in range(14):
#     plt.plot(x, nu[7,i,:],"r")
#     plt.show()

# nu=np.load("D:/aa_micro_expression/boshi/CAS(ME)2/lables_start_end.npy")
# nu=np.array(nu)
# print(nu.shape)
# x=np.arange(5000)
# for i in range(9):
#     plt.title(str(i))
#     plt.plot(x, nu[1,i,:],"r")
#     plt.show()
#
# def summ(ll):
#     pp=np.zeros(len(ll)+1)
#     for  i in range(0,len(ll)):
#         pp[i+1]=pp[i]+ll[i]
#     return pp
# for i in range(1,3):
#     print(i)


# import numpy as np
# from scipy.misc import derivative
# def f(x):
#     return x**5
# for x in range(1, 4):
#     print(derivative(f, x, dx=1e-6))

# path1="D:/dataset/micro_datatset/test_SAMM/006_1/200.npy"
# flow200=np.load(path1)
# width=200
# print(flow200.shape)
# print(flow200[16 * width:17 * width, ::])

# a = [1, 2, 3, 4, 1, 2, 3, 4]
# print(a[3])
# a=np.array(a)
# a=a.reshape((2,-1,2))
# print(a)

# pathp="D:/dataset/micro_datatset/test_SAMM_yolo/" +"006" +"/"# 存储的位置
# data11=np.load(pathp+"data.npy")
# # land11=np.load(pathp+"land.npy")
# print(data11.shape)
# print(land11.shape)
