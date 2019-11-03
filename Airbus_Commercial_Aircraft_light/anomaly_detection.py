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

# This file detects anomalies of Airbus Commercial Aircraft and specifies the parameter, the time to start, and the time that the anomaly ends. We show the result of anomalies.
# Trace des maximums de losses par fichiers
def trace_valid_anomalies_max_sum_loss_500(matrix, labels, filelist, predict, seuil_max_only=0.0001, seuil_sum_only=0.0005, seuil_max_com=0.0003, seuil_sum_com=0.00006):
    ########################
    # matrix: A serie of the input matrices with the corresponding shape
    # labels: The time labels correspond to the input matrix
    # filelist: The file name correspond to the matrices
    # predict: The output matrices of the input matrices names "matrix"
    # seuil_max_only: The thresholding of only maximum value
    # seuil_sum_only: The thresholding of only sum value
    # seuil_max_com: The thresholding of maximum value when together with the sum value
    # seuil_sum_com: The thresholding of sum value when together with the maximum value
    def most_frequent(List): 
        # To find the most frequent element in the list.
        counter = 0
        num = List[0] 
        for i in List: 
            curr_frequency = List.count(i) 
            if(curr_frequency> counter): 
                counter = curr_frequency 
                num = i 
        return num 
    #########################
    def diag_sum(matrix):
        # To find sum of all rows and columns of all diagonal values.
        res = np.zeros(matrix.shape[0])
        for i in range(matrix.shape[0]):
            res[i] = matrix[i,:].sum()+matrix[:,i].sum()-matrix[i,i]
        return res
    #########################
    # The main part of the function starts
    # The list "total_list" is the list that contains all anomalies per data ids.
    total_list = []
    # The list "loss" is the list of each anomaly.
    loss = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[3]):
            loss_value = ((matrix[i,:,:,j,0]-predict[i,:,:,j,0])**2).max()
            loss_value_sum = np.sum(((matrix[i,:,:,j,0]-predict[i,:,:,j,0])**2))
            label = int((labels[i,j,0]+labels[i,j,labels.shape[2]-1])/2.0)
            filename = filelist[i]
            loss.append([filename, label, loss_value, loss_value_sum, int(i), int(j)])
    sousmission = []
    new_file_list = list(set(filelist))
    dirs_valid = []
    for jj in range(len(new_file_list)):
        dirs_valid.append(new_file_list[jj][0:len(new_file_list[jj])-4])        
    for i in range(len(dirs_valid)):
        sep_list = []
        #print('len loss : '+ str(len(loss)))
        for j in range(len(loss)):
            if loss[j][0][0:len(loss[j][0])-4] == dirs_valid[i]:
                sep_list.append( [  loss[j][1], loss[j][2], loss[j][3], loss[j][4], loss[j][5], loss[j][6], loss[j][7]  ] )
                #label, loss_value, loss_value_sum, matrix[0], matrix[3]
                #print("done")
        loss2 = sorted(sep_list, key = lambda sep_list:sep_list[0] )
        loss3 = np.asarray(loss2)
        #print(loss3)
        if loss3.shape[0]>0:
            total_list.append(loss3)
            loss_list = []
            parameter = []
            shape = pd.read_csv(path_valid+'/'+dirs_valid[i], index_col=0).shape[0]
            for k in range(loss3.shape[0]):
                error_matrix = ((matrix[int(loss3[k,3]),:,:,int(loss3[k,4]),0]
                                -predict[int(loss3[k,3]),:,:,int(loss3[k,4]),0])**2)
                sum_loss = np.sum(error_matrix)
                if (loss3[k,1]>seuil_max_only or loss3[k,2]>seuil_sum_only) or (loss3[k,1]>seuil_max_com and loss3[k,2]>seuil_sum_com):
                    #add the time of anomalies
                    loss_list.append(loss3[k,0])
                    # compute the parameter which is anomalies
                    sum_first_list = np.zeros(len(data.params_of_interest))
                    for j in range(len(data.params_of_interest)):
                        #print("loss3[k,2] : "+str(loss3[k,2]))
                        sum_first_list[j] = np.sum(error_matrix, axis = 0)[j]+np.sum(error_matrix, axis = 1)[j]
                    parameter.append(data.params_of_interest[np.argmax(sum_first_list)])
            if loss_list: # if loss_list is not empty, then plot all anomaly and mark anomaly areas in red. 
                print("LOSS_LIST: ")
                print(loss_list)
                final_loss_time = [max(0,min(loss_list)-150), min(max(loss_list)+150, shape)]
                final_parameter = most_frequent(parameter)
                print("len(final_loss_time) : "+str(len(final_loss_time)))
                print("i: "+str(i))
                sousmission.append([dirs_valid[i], final_parameter, final_loss_time[0], final_loss_time[1]])    
                print("File : "+dirs_valid[i])
                print("Parameter : "+final_parameter)
                # Plot the anomalies founded in the maximum values of anomalies
                print("loss max : ")
                plt.plot(loss3[0:300,0], loss3[0:300,1])
                #plt.axvspan(final_loss_time[0], final_loss_time[1], color='red', alpha=0.5)
                plt.show()
                # Plot the anomalies founded in the diagonal sum values of anomalies                
                print("loss sum : ")
                plt.plot(loss3[0:300,0], loss3[0:300,2])
                #plt.axvspan(final_loss_time[0], final_loss_time[1], color='red', alpha=0.5)
                plt.show()
    # The result, total_list and sub-missions.
    return total_list, sousmission

if __name__ == "__main__":
    #Load test data
    ############################
    ########################################
    # The path_mat_valid1 is the path of validation data for the 1st flight and ground autoencoder, the path_lab_valid1 is the label of time correspond to the files of path_mat_valid1. And the dirs_mat_valid1 is the list of filenames of path_mat_valid1, and filenames of path_mat_valid1 are the same of path_lab_valid1.
    path_mat_valid1 = 'Data/valid_matrix_1'
    dirs_mat_valid1 = os.listdir(path_mat_valid1)
    path_lab_valid1 = 'Data/valid_labels_1'
    matrix_flight_valid1 = []
    matrix_ground_valid1 = []
    labels_flight_valid1 = []
    labels_ground_valid1 = []
    fileic_flight_valid1 = []
    fileic_ground_valid1 = []
    length=100
    print("Total number of files: "+str(len(dirs_mat_valid1)))
    # The filenames start of "B" are flight test, and the filenames start from "C" are ground test. In the training and test, we need to separate those data into different frameworks in order to guarantee the quality of result. The matrices correspond are registrated in list "matrix_flight_valid" or "matrix_ground_valid", the labels are registrated in list "labels_flight_valid" or "labels_ground_valid". And the filenames correspond are registrated in "fileic_flight_valid" or "fileic_ground_valid".
    for i in range(len(dirs_mat_valid1)):
        matrix = np.load(path_mat_valid1+'/'+dirs_mat_valid1[i])
        labels = np.load(path_lab_valid1+'/'+dirs_mat_valid1[i])
        if dirs_mat_valid1[i][0] == 'B':
            for j in range(matrix.shape[0]): 
                matrix_flight_valid1.append(matrix[j,:,:,:,:])
                labels_flight_valid1.append(labels[j,:,:])
                fileic_flight_valid1.append(dirs_mat_valid1[i])
        else:
            for j in range(matrix.shape[0]): 
                matrix_ground_valid1.append(matrix[j,:,:,:,:])
                labels_ground_valid1.append(labels[j,:,:])
                fileic_ground_valid1.append(dirs_mat_valid1[i])

        #print(str(i+1)+"e done!")
    ########################################
    # The path_mat_valid2 is the path of validation data for the 2nd flight and ground autoencoder, the path_lab_valid2 is the label of time correspond to the files of path_mat_valid2. And the dirs_mat_valid1 is the list of filenames of path_mat_valid2, and filenames of path_mat_valid2 are the same of path_lab_valid2.
    path_mat_valid2 = 'Data/valid_matrix_2'
    dirs_mat_valid2 = os.listdir(path_mat_valid2)
    path_lab_valid2 = 'Data/valid_labels_2'
    matrix_flight_valid2 = []
    matrix_ground_valid2 = []
    labels_flight_valid2 = []
    labels_ground_valid2 = []
    fileic_flight_valid2 = []
    fileic_ground_valid2 = []
    length=100
    print("Total number of files: "+str(len(dirs_mat_valid2)))
    # The filenames start of "B" are flight test, and the filenames start from "C" are ground test. In the training and test, we need to separate those data into different frameworks in order to guarantee the quality of result. The matrices correspond are registrated in list "matrix_flight_valid" or "matrix_ground_valid", the labels are registrated in list "labels_flight_valid" or "labels_ground_valid". And the filenames correspond are registrated in "fileic_flight_valid" or "fileic_ground_valid".
    for i in range(len(dirs_mat_valid2)):
        matrix = np.load(path_mat_valid2+'/'+dirs_mat_valid2[i])
        labels = np.load(path_lab_valid2+'/'+dirs_mat_valid2[i])
        if dirs_mat_valid2[i][0] == 'B':
            for j in range(matrix.shape[0]): 
                matrix_flight_valid2.append(matrix[j,:,:,:,:])
                labels_flight_valid2.append(labels[j,:,:])
                fileic_flight_valid2.append(dirs_mat_valid2[i])
        else:
            for j in range(matrix.shape[0]): 
                matrix_ground_valid2.append(matrix[j,:,:,:,:])
                labels_ground_valid2.append(labels[j,:,:])
                fileic_ground_valid2.append(dirs_mat_valid2[i])

        #print(str(i+1)+"e done!")
    ########################################
    # The path_mat_valid3 is the path of validation data for the 3rd flight and ground autoencoder, the path_lab_valid3 is the label of time correspond to the files of path_mat_valid3. And the dirs_mat_valid3 is the list of filenames of path_mat_valid3, and filenames of path_mat_valid3 are the same of path_lab_valid3.
    path_mat_valid3 = 'Data/valid_matrix_3'
    dirs_mat_valid3 = os.listdir(path_mat_valid3)
    path_lab_valid3 = 'Data/valid_labels_3'
    matrix_flight_valid3 = []
    matrix_ground_valid3 = []
    labels_flight_valid3 = []
    labels_ground_valid3 = []
    fileic_flight_valid3 = []
    fileic_ground_valid3 = []
    length=20
    print("Total number of files: "+str(len(dirs_mat_valid3)))
    # The filenames start of "B" are flight test, and the filenames start from "C" are ground test. In the training and test, we need to separate those data into different frameworks in order to guarantee the quality of result. The matrices correspond are registrated in list "matrix_flight_valid" or "matrix_ground_valid", the labels are registrated in list "labels_flight_valid" or "labels_ground_valid". And the filenames correspond are registrated in "fileic_flight_valid" or "fileic_ground_valid".
    for i in range(len(dirs_mat_valid3)):
        matrix = np.load(path_mat_valid3+'/'+dirs_mat_valid3[i])
        labels = np.load(path_lab_valid3+'/'+dirs_mat_valid3[i])
        if dirs_mat_valid3[i][0] == 'B':
            for j in range(matrix.shape[0]): 
                matrix_flight_valid3.append(matrix[j,:,:,:,:])
                labels_flight_valid3.append(labels[j,:,:])
                fileic_flight_valid3.append(dirs_mat_valid3[i])
        else:
            for j in range(matrix.shape[0]): 
                matrix_ground_valid3.append(matrix[j,:,:,:,:])
                labels_ground_valid3.append(labels[j,:,:])
                fileic_ground_valid3.append(dirs_mat_valid3[i])

        #print(str(i+1)+"e done!")
    ########################################
    # The path_mat_valid4 is the path of validation data for the 4th flight and ground autoencoder, the path_lab_valid4 is the label of time correspond to the files of path_mat_valid4. And the dirs_mat_valid1 is the list of filenames of path_mat_valid4, and filenames of path_mat_valid4 are the same of path_lab_valid4.
    path_mat_valid4 = 'Data/valid_matrix_4'
    dirs_mat_valid4 = os.listdir(path_mat_valid4)
    path_lab_valid4 = 'Data/valid_labels_4'
    matrix_flight_valid4 = []
    matrix_ground_valid4 = []
    labels_flight_valid4 = []
    labels_ground_valid4 = []
    fileic_flight_valid4 = []
    fileic_ground_valid4 = []
    length=20
    print("Total number of files: "+str(len(dirs_mat_valid4)))
    # The filenames start of "B" are flight test, and the filenames start from "C" are ground test. In the training and test, we need to separate those data into different frameworks in order to guarantee the quality of result. The matrices correspond are registrated in list "matrix_flight_valid" or "matrix_ground_valid", the labels are registrated in list "labels_flight_valid" or "labels_ground_valid". And the filenames correspond are registrated in "fileic_flight_valid" or "fileic_ground_valid".
    for i in range(len(dirs_mat_valid4)):
        matrix = np.load(path_mat_valid4+'/'+dirs_mat_valid4[i])
        labels = np.load(path_lab_valid4+'/'+dirs_mat_valid4[i])
        if dirs_mat_valid4[i][0] == 'B':
            for j in range(matrix.shape[0]): 
                matrix_flight_valid4.append(matrix[j,:,:,:,:])
                labels_flight_valid4.append(labels[j,:,:])
                fileic_flight_valid4.append(dirs_mat_valid4[i])
        else:
            for j in range(matrix.shape[0]): 
                matrix_ground_valid4.append(matrix[j,:,:,:,:])
                labels_ground_valid4.append(labels[j,:,:])
                fileic_ground_valid4.append(dirs_mat_valid4[i])

        #print(str(i+1)+"e done!")
    ########################################
    ###############################
    # As we know, the matrices correspond are registrated in list "matrix_flight_valid" or "matrix_ground_valid", the labels are registrated in list "labels_flight_valid" or "labels_ground_valid". And the filenames correspond are registrated in "fileic_flight_valid" or "fileic_ground_valid". We then tranform those lists into matrices in order to put them into the deep learning frameworks.
    X_test_flight1 = np.asarray(matrix_flight_valid1)
    X_test_ground1 = np.asarray(matrix_ground_valid1)
    X_test_label_flight1 = np.asarray(labels_flight_valid1)
    X_test_label_ground1 = np.asarray(labels_ground_valid1)
    print("1st flight test data shape = "+str(X_test_flight1.shape))
    print("1st flight test label shape = "+str(X_test_label_flight1.shape))
    print("1st flight file fileic shape = "+str(len(fileic_flight_valid1)))
    print("1st ground test data shape = "+str(X_test_ground1.shape))
    print("1st ground test label shape = "+str(X_test_label_ground1.shape))
    print("1st ground file fileic shape = "+str(len(fileic_ground_valid1)))
    ###############################
    X_test_flight2 = np.asarray(matrix_flight_valid2)
    X_test_ground2 = np.asarray(matrix_ground_valid2)
    X_test_label_flight2 = np.asarray(labels_flight_valid2)
    X_test_label_ground2 = np.asarray(labels_ground_valid2)
    print("2nd flight test data shape = "+str(X_test_flight2.shape))
    print("2nd flight test label shape = "+str(X_test_label_flight2.shape))
    print("2nd flight file fileic shape = "+str(len(fileic_flight_valid2)))
    print("2nd ground test data shape = "+str(X_test_ground2.shape))
    print("2nd ground test label shape = "+str(X_test_label_ground2.shape))
    print("2nd ground file fileic shape = "+str(len(fileic_ground_valid2)))
    ###############################
    X_test_flight3 = np.asarray(matrix_flight_valid3)
    X_test_ground3 = np.asarray(matrix_ground_valid3)
    X_test_label_flight3 = np.asarray(labels_flight_valid3)
    X_test_label_ground3 = np.asarray(labels_ground_valid3)
    print("3rd flight test data shape = "+str(X_test_flight3.shape))
    print("3rd flight test label shape = "+str(X_test_label_flight3.shape))
    print("3rd flight file fileic shape = "+str(len(fileic_flight_valid3)))
    print("3rd ground test data shape = "+str(X_test_ground3.shape))
    print("3rd ground test label shape = "+str(X_test_label_ground3.shape))
    print("3rd ground file fileic shape = "+str(len(fileic_ground_valid3)))
    ###############################
    X_test_flight4 = np.asarray(matrix_flight_valid4)
    X_test_ground4 = np.asarray(matrix_ground_valid4)
    X_test_label_flight4 = np.asarray(labels_flight_valid4)
    X_test_label_ground4 = np.asarray(labels_ground_valid4)
    print("4th flight test data shape = "+str(X_test_flight4.shape))
    print("4th flight test label shape = "+str(X_test_label_flight4.shape))
    print("4th flight file fileic shape = "+str(len(fileic_flight_valid4)))
    print("4th ground test data shape = "+str(X_test_ground4.shape))
    print("4th ground test label shape = "+str(X_test_label_ground4.shape))
    print("4th ground file fileic shape = "+str(len(fileic_ground_valid4)))
    print("########################")
    ############################### 

    # Once all validation data are transformed into matrices, we load all pre-trained model from file "train.py" and then compile those model.
    ############################
    # load json and create model
    json_file_ground = open(path+'/model_ground1.json', 'r')
    loaded_model_json_ground1 = json_file_ground.read()
    json_file_ground.close()
    autoencoder_ground1 = model_from_json(loaded_model_json_ground1)
    # load weights into new model
    autoencoder_ground1.load_weights(path+'/model_ground1.h5')
    print("Loaded ground model 1 from disk")
    autoencoder_ground1.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # load json and create model
    json_file_flight = open(path+'/model_flight1.json', 'r')
    loaded_model_json_flight1 = json_file_flight.read()
    json_file_flight.close()
    autoencoder_flight1 = model_from_json(loaded_model_json_flight1)
    # load weights into new model
    autoencoder_flight1.load_weights(path+'/model_flight1.h5')
    print("Loaded flight model 1 from disk")
    autoencoder_flight1.compile(optimizer='adam', loss='mse', metrics=['mae'])

    ############################
    # load json and create model
    json_file_ground = open(path+'/model_ground2.json', 'r')
    loaded_model_json_ground2 = json_file_ground.read()
    json_file_ground.close()
    autoencoder_ground2 = model_from_json(loaded_model_json_ground2)
    # load weights into new model
    autoencoder_ground2.load_weights(path+'/model_ground2.h5')
    print("Loaded ground model 2 from disk")
    autoencoder_ground2.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # load json and create model
    json_file_flight = open(path+'/model_flight2.json', 'r')
    loaded_model_json_flight2 = json_file_flight.read()
    json_file_flight.close()
    autoencoder_flight2 = model_from_json(loaded_model_json_flight2)
    # load weights into new model
    autoencoder_flight2.load_weights(path+'/model_flight2.h5')
    print("Loaded flight model 2 from disk")
    autoencoder_flight2.compile(optimizer='adam', loss='mse', metrics=['mae'])

    ############################
    # load json and create model
    json_file_ground = open(path+'/model_ground3.json', 'r')
    loaded_model_json_ground3 = json_file_ground.read()
    json_file_ground.close()
    autoencoder_ground3 = model_from_json(loaded_model_json_ground3)
    # load weights into new model
    autoencoder_ground3.load_weights(path+'/model_ground3.h5')
    print("Loaded ground model 3 from disk")
    autoencoder_ground3.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # load json and create model
    json_file_flight = open(path+'/model_flight3.json', 'r')
    loaded_model_json_flight3 = json_file_flight.read()
    json_file_flight.close()
    autoencoder_flight3 = model_from_json(loaded_model_json_flight3)
    # load weights into new model
    autoencoder_flight3.load_weights(path+'/model_flight3.h5')
    print("Loaded flight model 3 from disk")
    autoencoder_flight3.compile(optimizer='adam', loss='mse', metrics=['mae'])

    ############################
    # load json and create model
    json_file_ground = open(path+'/model_ground4.json', 'r')
    loaded_model_json_ground4 = json_file_ground.read()
    json_file_ground.close()
    autoencoder_ground4 = model_from_json(loaded_model_json_ground4)
    # load weights into new model
    autoencoder_ground4.load_weights(path+'/model_ground4.h5')
    print("Loaded ground model 4 from disk")
    autoencoder_ground4.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # load json and create model
    json_file_flight = open(path+'/model_flight4.json', 'r')
    loaded_model_json_flight4 = json_file_flight.read()
    json_file_flight.close()
    autoencoder_flight4 = model_from_json(loaded_model_json_flight4)
    # load weights into new model
    autoencoder_flight4.load_weights(path+'/model_flight4.h5')
    print("Loaded flight model 4 from disk")
    autoencoder_flight4.compile(optimizer='adam', loss='mse', metrics=['mae'])

    
    # After loading and compiling models, we put the matrices of validation datasets into the previous models and launch the prediction. The result "X_flight_pred" and "X_ground_pred" are the result of prediction of "X_test_flight" and "X_test_ground".
    #########
    X_flight_pred1 = autoencoder_flight1.predict(X_test_flight1)
    X_ground_pred1 = autoencoder_ground1.predict(X_test_ground1)
    #########
    X_flight_pred2 = autoencoder_flight2.predict(X_test_flight2)
    X_ground_pred2 = autoencoder_ground2.predict(X_test_ground2)
    #########
    X_flight_pred3 = autoencoder_flight3.predict(X_test_flight3)
    X_ground_pred3 = autoencoder_ground3.predict(X_test_ground3)
    #########
    X_flight_pred4 = autoencoder_flight4.predict(X_test_flight4)
    X_ground_pred4 = autoencoder_ground4.predict(X_test_ground4)

    ################################################################
    # This is the function that plots the maximum and diagonal sum of loss into time sequences and mark the anomaly moments and anomaly parameters.
    list3, sousmission3 = trace_valid_anomalies_max_sum_loss_500(X_test_ground4, X_test_label_ground4, fileic_ground_valid4, X_ground_pred4)
