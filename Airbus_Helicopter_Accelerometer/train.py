import numpy as np
import matplotlib.pyplot as plt
import data
import wpe
from colorama import Fore, Back, Style 

print("------------- Computing Wavelet Packet Energy. -------------")
# Tranform training data and test data from .h5 files into signals
X_train, X_test = data.file_to_signals('Data','dftrain.h5'), data.file_to_signals('Data','dffinal.h5')
X_train_wpe2 = wpe.wpe3(X_train)
X_test_wpe2  = wpe.wpe3(X_test )
# Read the training data, validation data and test data from file "dftrain.h5", "dfvalid.h5" and "dffinal.h5" and name them as "df3", "dv3", "dt3".
df3 = data.get_raw_data('Data','dftrain.h5')
dv3 = data.get_raw_data('Data','dfvalid.h5')
dt3 = data.get_raw_data('Data','dffinal.h5')

# Computing the mean and sigma of ratios between the 1st column to the other columns.
def mean_sigma(matrix):
    output = np.zeros((matrix.shape[0],matrix.shape[1]-1))
    mu = np.zeros(matrix.shape[1]-1)
    sigma = np.zeros(matrix.shape[1]-1)
    for j in range(matrix.shape[1]-1):
        for i in range(matrix.shape[0]):
            if matrix[i,0]!=0:
                output[i,j]=matrix[i,j+1]/matrix[i,0]
        mu[j]=np.mean(output[:,j])
        sigma[j] = np.var(output[:,j])
    return mu, sigma

# The function of anomaly detection.  
def mu_sigma_anomalies(train_data, test_data, k_vector, List, print_normal=True, plot = True):
    # The 3 sub functions to finetune the value of k.
    def Compute_k_vector(matrix, number_of_min):
        mu, sigma = mean_sigma(matrix)
        epsilon   = min_vector(matrix)
        k = np.zeros(mu.shape[0])
        for i in range(mu.shape[0]):
            k[i] = (mu[i]-number_of_min*epsilon[i])/sigma[i]
        return k

    def Compute_k_vector2(mu, sigma, matrix, n):
        k = np.zeros(mu.shape[0])
        for i in range(mu.shape[0]):
            k[i] = (mu[i]-matrix[n,i+1]/matrix[n,0])/sigma[i]
        return k
        
    def Compute_min_k(mu, sigma, matrix, anomalies_array):
        k = Compute_k_vector2(mu, sigma, matrix, anomalies_array[0])
        for i in range(anomalies_array.shape[0]-1):
            k_min = Compute_k_vector2(mu, sigma, matrix, anomalies_array[i+1])
            for j in range(k.shape[0]):
                if k_min[j]<k[j]:
                    k[j] = k_min[j]
        return k
    # The main part of the function begins
    print("-------------    Anomaly detection process.    -------------")
    mu_train, sigma_train = mean_sigma(train_data) # Computing the mu and sigma of ratios of training data
    mu_test , sigma_test  = mean_sigma(test_data) # Computing the mu and sigma of ratios of test data
    anomalies=[] # The list of anomalies
    k = [] # The value of k
    sousmission=np.zeros((test_data.shape[0],2))
    for i in range(test_data.shape[0]):
        sousmission[i,0]=int(i)
        bool=True
        for j in List:
            if (test_data[i,j]/test_data[i,0])>mu_train[j-1]-k_vector[j-1]*sigma_train[j-1]:
                bool=False
                break
        if test_data[i, 0]==0 and test_data[i, 1]==0:
            bool=True
        if bool==True:
            anomalies.append(int(i))
            sousmission[i,1]=int(1)
            if plot==True:
                print("The sample "+str(i)+" is anomalie")
                print("Original signaux: ")
                plt.plot(dt3[i,:])
                plt.show()
                print("Extraction: ")
                plt.bar(range(len(test_data[1,:])), test_data[i,:], fc = 'r')
                plt.show()
                print("k : " + str(Compute_k_vector2(mu_train, sigma_train, test_data, i)))
        else:
            if print_normal:
                sousmission[i,1]=int(0)
                if plot==True:
                    print(Fore.RED + "Sample "+str(i)+" not anomalie")
                    print("Original signaux: ")
                    plt.plot(dv3[i,:])
                    plt.show()
                    print("Extraction: ")
                    plt.bar(range(len(test_data[1,:])), test_data[i,:], fc = 'r')
                    plt.show()
                    print("k : " + str(Compute_k_vector2(mu_train, sigma_train, test_data, i)))
    anomalies_array=np.asarray(anomalies)
    len_anom = anomalies_array.shape[0]
    return anomalies_array, sousmission, len_anom


        

List =[  2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
k2 = [-0.09000000,  0.13033350,  -0.1000000,  0.05535343,  0.07326000,  0.23500000,
       0.18322308,  0.01006837,  0.20263449,  0.07664112,  0.29845937,  0.13008332,
       0.12567191,  0.08813385,  0.07928000,  7.01902489,  0.66661028,  0.88209162,
       0.63641430,  0.58095742,  0.80709155,  0.43930355,  0.90342995,  0.09409717,
       0.15596115,  0.03365090,  0.13584499,  0.56785683,  0.34207958,  0.07809385,
       0.13497834]

if __name__ == "__main__":

    anomalies_array2, sousmission2, len_anom2 = mu_sigma_anomalies(X_train_wpe2, X_test_wpe2, k2, List, print_normal=True, plot = False)
    print("len_anom2 : "+str(len_anom2))
    np.savetxt('result'+'/+'"sousmission2.csv", sousmission2, delimiter=",", header = "seqID,anomaly", fmt="%i", comments = '')
