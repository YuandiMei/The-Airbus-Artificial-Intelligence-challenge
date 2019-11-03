import data
import convlstm_autoencoder
import train
import numpy as np
from colorama import Fore, Back, Style
import tensorflow as tf
keras=tf.contrib.keras
l2=keras.regularizers.l2
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.models import Model, model_from_json, Sequential
from keras.callbacks import History 
from keras import losses, optimizers
from keras.utils import plot_model
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

# This file detects anomalies of Airbus Defence and Space Antenna and specifies the parameter, the time to start, and the time that the anomaly ends. We show the result of anomalies.
def plot_matrix_test(id_list_of_file, matrix_ini, matrix_rec, only_anomaly=True):
    # id_list_of_file: The list of id of the time sequences 
    # matrix_ini: The initial signature matrices
    # matrix_rec: The output matrices reconstructed by the autoencoder from the initial matrices "matrix_ini"
    # only_anomaly: parameter that defines if we plot only anomalies or not
    for i in range(len(id_list_of_file)):
        for j in range((matrix_ini.shape[3])):
            # Compute the mean square error matrices
            error_matrix = ((matrix_ini[i,:,:,j,0] - matrix_rec[i,:,:,j,0])**2)
            if anomaly_list[list(anomaly_list[:,0]).index(id_list_of_file[i]), 1] == 1:
                # Plot the initial and reconstructed matrices, mark the id of matrices in red.
                print(Fore.RED + "signaux {0} anomaly".format(id_list_of_file[i]))
                print(Fore.RED + "Initial matrix: ")
                plt.imshow(matrix_ini[i,:,:,j,0], cmap=plt.get_cmap('binary'))
                plt.show()
                print(Fore.RED + "Reconstructed matrix: ")
                plt.imshow(matrix_rec[i,:,:,j,0], cmap=plt.get_cmap('binary'))
                plt.show()
                #Plot the mean square error matrix of the initial and reconstructed matrices above.
                print(Fore.RED + "mse loss matrix: ")
                plt.imshow(error_matrix, cmap=plt.get_cmap('binary'))
                plt.show()
            else:
                # Execute this part only if when we do not recommand "only_anomaly=True"
                if not(only_anomaly):
                    print("signaux {0} not anomaly".format(id_list_of_file[i]))
                    print("Initial matrix: ")
                    plt.imshow(matrix_ini[i,:,:,j,0], cmap=plt.get_cmap('binary'))
                    plt.show()
                    print("Reconstructed matrix: ")
                    plt.imshow(matrix_rec[i,:,:,j,0], cmap=plt.get_cmap('binary'))
                    plt.show()
                    print("mse loss matrix: ")
                    plt.imshow(error_matrix, cmap=plt.get_cmap('binary'))
                    plt.show()

# This file plots the time sequences of maximum loss and sum of loss.                
def plot_loss_test(id_list_of_file, matrix_ini, matrix_rec, labels, only_anomaly=True):
    # id_list_of_file: The list of id of the time sequences 
    # matrix_ini: The initial signature matrices
    # matrix_rec: The output matrices reconstructed by the autoencoder from the initial matrices "matrix_ini"
    # only_anomaly: parameter that defines if we plot only anomalies or not
    for i in range(len(id_list_of_file)):
        Max_error = []
        Sum_error = []
        Tim_error = []
        # Get the max loss and sum loss list
        for j in range((matrix_ini.shape[3])):
            error_matrix = (matrix_ini[i,:,:,j,0] - matrix_rec[i,:,:,j,0])**2
            max_error = error_matrix.max()
            sum_error = np.sum(error_matrix)
            Max_error.append(max_error)
            Sum_error.append(sum_error)
            Tim_error.append((labels[i,j,0]+labels[i,j,labels.shape[2]-1])/2)   
        # Plot the maximum value of signature matrix depending on time labels
        if anomaly_list[list(anomaly_list[:,0]).index(id_list_of_file[i]), 1] == 1:
            print(Fore.RED + "signaux {0} anomaly".format(id_list_of_file[i]))
            print(Fore.RED + "Maximum loss: ")
            print(Max_error)
            plt.plot(Tim_error, Max_error)
            plt.show()
            print(Fore.RED + "Somme de loss: ")
            print(Sum_error)
            plt.plot(Tim_error, Sum_error)
            plt.show()
        else:
            # Plot only if we setted "only_anomaly=False" 
            if not(only_anomaly):
                print("signaux {0} not anomaly".format(id_list_of_file[i]))
                print("signaux {0} anomaly".format(id_list_of_file[i]))
                print("Maximum loss: ")
                print(Max_error)
                plt.plot(Tim_error, Max_error)
                plt.show()
                print("Somme de loss: ")
                print(Sum_error)
                plt.plot(Tim_error, Sum_error)
                plt.show()
                
if __name__ == '__main__':

    #Load model
    #load autoencoder3
    #############################################
    path="Models"
    json_file = open(path+"/"+'model3.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    autoencoder3 = model_from_json(loaded_model_json)
    # load weights into new model
    autoencoder3.load_weights(path+"/"+"model3.h5")
    print("Loaded model from disk")
    autoencoder3.compile(optimizer='adam', loss='mse', metrics=['mae'])
    #load autoencoder4
    #############################################
    json_file = open(path+"/"+'model4.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    autoencoder4 = model_from_json(loaded_model_json)
    # load weights into new model
    autoencoder4.load_weights(path+"/"+"model4.h5")
    print("Loaded model from disk")
    autoencoder4.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Load test data
    ########################################
    # The test_mat1 is the path of validation data for the 1st autoencoder from test dataset, the test_lab1 is the label of time correspond to the files of test_mat1. The vali_mat1 is the path of validation data for the 1st autoencoder from validation dataset, the vali_lab1 is the label of time correspond to the files of vali_mat1.
    path="Data"   
    # The function that loads matrices, the time labels correspond, and the ids correspond to the matrices.
    def load_matrix(path_mat, path_lab):
        dirs = os.listdir(path_mat)
        matrix5 = []
        labels5 = []
        idlist5 = []
        for i in range(len(dirs)):
            matrix = np.load(path_mat+'/'+dirs[i])
            labels = np.load(path_lab+'/'+dirs[i])
            idname = int(dirs[i][0:len(dirs[i])-4])
            matrix5.append(matrix[0,:,:,:,:])
            labels5.append(labels[0,:,:])
            idlist5.append(idname)
        return np.asarray(matrix5), np.asarray(labels5), np.asarray(idlist5)
    #############################################
    test_mat1 = path + '/validation_matrix1'
    test_lab1 = path + '/validation_labels1'
    vali_mat1 = path + '/real_validation1'
    vali_lab1 = path + '/real_validlabel1'
    ############
    X_train1, X_train_lab1, id_train1 = load_matrix(path_mat1, path_lab1)
    X_test1 , X_test_lab1 , id_test1  = load_matrix(test_mat1, test_lab1)
    X_valid1, X_valid_lab1, id_valid1 = load_matrix(vali_mat1, vali_lab1)
    print("Shape of X_test1:  "+ str(X_test1.shape) )
    print("Shape of X_valid1: "+ str(X_valid1.shape))
    print("Shape of X_test_lab1:  "+ str(X_test_lab1.shape) )
    print("Shape of X_valid_lab1: "+ str(X_valid_lab1.shape))
    print("Shape of id_test1:  "+ str(id_test1.shape) )
    print("Shape of id_valid1: "+ str(id_valid1.shape))

    ##############################################
    test_mat2 = path + '/validation_matrix2'
    test_lab2 = path + '/validation_labels2'
    vali_mat2 = path + '/real_validation2'
    vali_lab2 = path + '/real_validlabel2'
    ############
    X_test2 , X_test_lab2 , id_test2  = load_matrix(test_mat2, test_lab2)
    X_valid2, X_valid_lab2, id_valid2 = load_matrix(vali_mat2, vali_lab2)
    print("Shape of X_test2:  "+ str(X_test2.shape) )
    print("Shape of X_valid2: "+ str(X_valid2.shape))
    print("Shape of X_test_lab2:  "+ str(X_test_lab2.shape) )
    print("Shape of X_valid_lab2: "+ str(X_valid_lab2.shape))
    print("Shape of id_test2:  "+ str(id_test2.shape) )
    print("Shape of id_valid2: "+ str(id_valid2.shape))
    #Compute 26 parameters list
    def get_list(data):
        List1=[]
        List2=[]
        for i in range(data.shape[0]):
            for j in range(data.shape[3]):
                y = np.nonzero(data[i,:,:,j,0])
                List1.append(list(set(y[0])))
                List2.append(list(set(y[1])))
        return List1, List2
    List3, List4 = get_list(X_test1)
    List5, List6 = get_list(X_valid1)
    
    # The function commomElements is to find the commom elements of a group of lists
    def commonElements(arr): 
        result = set(arr[0]) 
        for currSet in arr[1:]: 
            result.union(currSet) 
        return list(result) 
    
    common_list3 = commonElements(List3)
    print("length 3: " + str(len(common_list3)))
    common_list4 = commonElements(List4)
    print("length 4: " + str(len(common_list4)))
    common_list5 = commonElements(List5)
    print("length 3: " + str(len(common_list5)))
    common_list6 = commonElements(List6)
    print("length 4: " + str(len(common_list6)))
    # The big_list is the list that contains only zeros in the signature matrices
    big_list=[]
    big_list.append(common_list1)
    big_list.append(common_list2)
    big_list.append(common_list3)
    big_list.append(common_list4)
    big_list.append(common_list5)
    big_list.append(common_list6)
    common_list = commonElements(big_list)
    print("Final length: " + str(len(common_list)))
    # The anti_common_list is the list of all elements except the big_list
    anti_common_list=[]
    for i in range(35):
        if not(i in common_list):
            anti_common_list.append(i)
    anti_common_list

    # Get the final matrix
    #########################
    Xtest1 = np.delete(X_test1, tuple(anti_common_list),axis=1)
    Xtest1 = np.delete(Xtest1, tuple(anti_common_list),axis=2)
    print("New Xtest1 shape: " + str(Xtest1.shape))
    Xtest2 = np.delete(X_test2, tuple(anti_common_list),axis=1)
    Xtest2 = np.delete(Xtest2, tuple(anti_common_list),axis=2)
    print("New Xtest2 shape: " + str(Xtest2.shape))
    #########################
    Xvalid1 = np.delete(X_valid1, tuple(anti_common_list),axis=1)
    Xvalid1 = np.delete(Xvalid1, tuple(anti_common_list),axis=2)
    print("New Xvalid1 shape: " + str(Xvalid1.shape))
    Xvalid2 = np.delete(X_valid2, tuple(anti_common_list),axis=1)
    Xvalid2 = np.delete(Xvalid2, tuple(anti_common_list),axis=2)
    print("New Xvalid2 shape: " + str(Xvalid2.shape))
    #########################

    # Normalizing matrix to improve the training result of the autoencoder
    def normalizing(five_D_matrix): 
        new_matrix = np.zeros((((( five_D_matrix.shape[0], five_D_matrix.shape[1], five_D_matrix.shape[2], five_D_matrix.shape[3], five_D_matrix.shape[4] )))))
        for i in range( five_D_matrix.shape[0] ):
            max_value = five_D_matrix[i,:,:,:,0].max()
            if max_value!=0:
                new_matrix[i,:,:,:,0] = five_D_matrix[i,:,:,:,0]/max_value
        return new_matrix
    #######################
    Xtest_norm1 = normalizing(Xtest1)
    Xtest_norm2 = normalizing(Xtest2)
    Xvalid_norm1 = normalizing(Xvalid1)
    Xvalid_norm2 = normalizing(Xvalid2)

    ##########################################################
    # Prediction of test and valid model
    # Test dataset prediction
    X_predict_test_norm1  = autoencoder4.predict(Xtest_norm1)
    X_predict_test_norm2  = autoencoder3.predict(Xtest_norm2)
    # valid dataset prediction
    X_predict_valid_norm1 = autoencoder4.predict(Xvalid_norm1)
    X_predict_valid_norm2 = autoencoder3.predict(Xvalid_norm2)

    ##########################################################
    # Test dataset prediction
    X_predict_test1  = autoencoder4.predict(Xtest1)
    X_predict_test2  = autoencoder3.predict(Xtest2)
    # valid dataset prediction
    X_predict_valid1 = autoencoder4.predict(Xvalid1)
    X_predict_valid2 = autoencoder3.predict(Xvalid2)
    
    # Plot all Error matrices of test data in the ConvLSTM Autoencoder
    plot_matrix_test(id_test1, Xtest1, X_predict_test1, only_anomaly=False)
    # Plot the maximum and the sum values depending on time of all time sequences in the test data.
    plot_loss_test(id_test1, Xtest1, X_predict_test1, X_test_lab1, only_anomaly=False)
