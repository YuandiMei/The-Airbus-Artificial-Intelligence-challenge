import pywt
import numpy as np

# Computing the Wavelet Packet Energy from each coefficient of Wavelet Packet Decomposition
def wpe(data):
    energy = 0
    for i in range(data.shape[0]):
        energy = energy+data[i]**2
    return energy

# Employing the Wavelet Packet Decomposition(WPD) into 16 sub-bands and computing the Wavelet Packet Decomposition(WPE) of each sub-band.
def wpe2(matrix):
    t_wpe = np.zeros((matrix.shape[0], 16))
    # Step1: The Wavelet Packet Decomposition using the Daubechies wavelet in mode symmetric.
    for i in range(matrix.shape[0]):
        wp = pywt.WaveletPacket(data=matrix[i], wavelet='db1', mode='symmetric')

        # Step2: Computing the Wavelet Packet Energy of each of 16 Wavelet Packet Decomposition sub-bands.
        t_wpe[i, 0] = wpe(wp['aaaa'].data)
        t_wpe[i, 1] = wpe(wp['aaad'].data)
        t_wpe[i, 2] = wpe(wp['aada'].data)
        t_wpe[i, 3] = wpe(wp['aadd'].data)

        t_wpe[i, 4] = wpe(wp['adaa'].data)
        t_wpe[i, 5] = wpe(wp['adad'].data)
        t_wpe[i, 6] = wpe(wp['adda'].data)
        t_wpe[i, 7] = wpe(wp['addd'].data)

        t_wpe[i, 8] = wpe(wp['daaa'].data)
        t_wpe[i, 9] = wpe(wp['daad'].data)
        t_wpe[i,10] = wpe(wp['dada'].data)
        t_wpe[i,11] = wpe(wp['dadd'].data)

        t_wpe[i,12] = wpe(wp['ddaa'].data)
        t_wpe[i,13] = wpe(wp['ddad'].data)
        t_wpe[i,14] = wpe(wp['ddda'].data)
        t_wpe[i,15] = wpe(wp['dddd'].data)

    return t_wpe

# Employing the Wavelet Packet Decomposition(WPD) into 32 sub-bands and computing the Wavelet Packet Decomposition(WPE) of each sub-band.
def wpe3(matrix):
    t_wpe = np.zeros((matrix.shape[0], 32))
    # Step1: The Wavelet Packet Decomposition using the Daubechies wavelet in mode symmetric.
    for i in range(matrix.shape[0]):
        wp = pywt.WaveletPacket(data=matrix[i], wavelet='db1', mode='symmetric')
        # Step2: Computing the Wavelet Packet Energy of each of 32 Wavelet Packet Decomposition sub-bands.
        t_wpe[i, 0] = wpe(wp['aaaaa'].data)
        t_wpe[i, 1] = wpe(wp['aaaad'].data)
        t_wpe[i, 2] = wpe(wp['aaada'].data)
        t_wpe[i, 3] = wpe(wp['aaadd'].data)
        t_wpe[i, 4] = wpe(wp['aadaa'].data)
        t_wpe[i, 5] = wpe(wp['aadad'].data)
        t_wpe[i, 6] = wpe(wp['aadda'].data)
        t_wpe[i, 7] = wpe(wp['aaddd'].data)
        t_wpe[i, 8] = wpe(wp['adaaa'].data)
        t_wpe[i, 9] = wpe(wp['adaad'].data)
        t_wpe[i,10] = wpe(wp['adada'].data)
        t_wpe[i,11] = wpe(wp['adadd'].data)
        t_wpe[i,12] = wpe(wp['addaa'].data)
        t_wpe[i,13] = wpe(wp['addad'].data)
        t_wpe[i,14] = wpe(wp['addda'].data)
        t_wpe[i,15] = wpe(wp['adddd'].data)
        t_wpe[i,16] = wpe(wp['daaaa'].data)
        t_wpe[i,17] = wpe(wp['daaad'].data)
        t_wpe[i,18] = wpe(wp['daada'].data)
        t_wpe[i,19] = wpe(wp['daadd'].data)
        t_wpe[i,20] = wpe(wp['dadaa'].data)
        t_wpe[i,21] = wpe(wp['dadad'].data)
        t_wpe[i,22] = wpe(wp['dadda'].data)
        t_wpe[i,23] = wpe(wp['daddd'].data)
        t_wpe[i,24] = wpe(wp['ddaaa'].data)
        t_wpe[i,25] = wpe(wp['ddaad'].data)
        t_wpe[i,26] = wpe(wp['ddada'].data)
        t_wpe[i,27] = wpe(wp['ddadd'].data)
        t_wpe[i,28] = wpe(wp['dddaa'].data)
        t_wpe[i,29] = wpe(wp['dddad'].data)
        t_wpe[i,30] = wpe(wp['dddda'].data)
        t_wpe[i,31] = wpe(wp['ddddd'].data)
    return t_wpe



