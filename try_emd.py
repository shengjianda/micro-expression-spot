
import numpy  as np
import pylab as plt
from PyEMD import EMD


# Define signal
t = np.linspace(0, 1, 200)
s = np.cos(11*2*np.pi*t*t) + 6*t*t
# def the_emd(s):
#     # Execute EMD on signal
#     t=np.arange(len(s))
#     IMF = EMD().emd(s,t)
#     N = IMF.shape[0]+1
#
#     # Plot results
#     plt.subplot(N,1,1)
#     print(N)
#     plt.plot(t, s, 'r')
#     plt.title("Input signal: $S(t)=cos(22\pi t^2) + 6t^2$")
#     plt.xlabel("Time [s]")
#
#     for n, imf in enumerate(IMF):
#         plt.subplot(N,1,n+2)
#         plt.plot(t, imf, 'g')
#         plt.title("IMF "+str(n+1))
#         plt.xlabel("Time [s]")
#
#
#     plt.tight_layout()
#     plt.savefig('simple_example')
#     plt.show()


def the_emd1(yuan, s, path, xuhao, frame_stride=1):
    # Execute EMD on signal
    t=np.arange(len(s) / frame_stride)
    s=np.array(s)
    IMF = EMD().emd(s, t)
    N = IMF.shape[0]

    # Plot results
    # if (int(xuhao) < 0):
    #     plt.subplot(2,1,1)
    #     plt.plot(t, yuan, 'r')
    #     plt.title("Input signal"+path+xuhao)
    #     plt.xlabel("Time [s]")

    imf_sum=np.zeros(IMF.shape[1])
    imf_sum1=np.zeros(IMF.shape[1])
    for n, imf in enumerate(IMF):
        if( n!=N-1):
            imf_sum1 = np.add(imf_sum1, imf)
        if(n!=0 ):
            imf_sum=np.add(imf_sum,imf)
    # if (int(xuhao) <0 ):
    #     plt.subplot(2,1,2)
    #     plt.plot(t, s, 'g')
    #     plt.title("IMF ")
    #     plt.xlabel("Time [s]")
    #
    #     plt.show()
    # plt.subplots(2,1)
    # plt.subplot(2,1,1)
    # plt.plot(imf_sum)
    # plt.subplot(2, 1, 2)
    # plt.plot(imf_sum1)
    # plt.show()

    return imf_sum,imf_sum1
