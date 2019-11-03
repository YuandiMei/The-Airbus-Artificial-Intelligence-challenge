import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import data

'''
This file computes the maximum loss and diagonal sum loss of loss matrices.
Those functions works on the known anomalies data in the training data in order to test the effect of method.
If the result of this file's function works well, then we employ the same method on the test data in file anomaly_detection.py.
'''

def max_sum_loss(model, matrix, label, w):
    # model: The autoencoder model
    # matrix: The input matrices
    # label: The time labels correspond to the matrix
    # w: The number of point to be computed in each signature matrix
    mini = 100
    loss = []
    Dia = []
    Dia_sum = []
    ##############################
    #Finding the most frequent element of a list
    def most_frequent(List): 
        counter = 0
        num = List[0] 
        for i in List: 
            curr_frequency = List.count(i) 
            if(curr_frequency> counter): 
                counter = curr_frequency 
                num = i 
        return num 
    #Computing the sum of each row and column combined
    def diag_sum(matrix):
        res = np.zeros(matrix.shape[0])
        for i in range(res.shape[0]):
            res[i] = np.sum(matrix[i,:])+np.sum(matrix[:,i])-matrix[i,i]
        return res
    ##############################
    #Computing the maximum, sum of all, sum of row and cloumn combined of each MSE loss matrix. 
    for k in range(len(dirs1_anomalies[14:])):
        loss1 = []
        X_flight_pred = model.predict(matrix[k])
        for i in range(X_flight_pred.shape[0]):
            for j in range(X_flight_pred.shape[3]):
                mat1 = np.reshape(matrix[k][i,:,:,j,:], (len(data.params_of_interest), len(data.params_of_interest)))
                mat2 = np.reshape(X_flight_pred[i,:,:,j,:],(len(data.params_of_interest), len(data.params_of_interest)))
                new_matrix = ((mat1 - mat2)**2)
                dia     = np.diag(new_matrix)
                dia_sum = diag_sum(new_matrix)
                
                index_max = np.argmax(dia)
                index_sum = np.argmax(dia_sum)
                loss1.append([int((label[k][i,j,0]+label[k][i,j,w-1])/2), index_max, 
                              index_sum, dia, dia_sum, new_matrix.max()])
        
        # Selecting the values that correspond to the parameter of anomaly
        loss2 = np.asarray(loss1)
        max_param_index = most_frequent(loss2[:,1].tolist())
        sum_param_index = most_frequent(loss2[:,2].tolist())
        loss3 = []
        for i in range(loss2.shape[0]):
            loss3.append([loss2[i,0], max_param_index, sum_param_index, loss2[i,3][max_param_index], 
                          loss2[i,4][sum_param_index], loss2[i,5]])
            Dia.append(loss2[i,3])
            Dia_sum.append(loss2[i,4])
        loss.append(np.asarray(loss3))
    return loss, Dia, Dia_sum


def show_anomalies_basic(loss):
    # Showing anomalies parameter and location via maximum loss function and sum loss function
    for k in range(len(dirs1_anomalies)):
        print("File {0}:".format(dirs1_anomalies[k]))
        df = pd.read_csv(path1_anomalies+'/'+dirs1_anomalies[k], index_col=0)
        print("Real anomaly parameter : " + anomalies_array[k][0])
        print("loss[{0}][0,1] : {1}".format(k,data.params_of_interest[int(loss[k][0,1])]))
        if anomalies_array[k][0] == data.params_of_interest[int(loss[k][0,1])]:
            print("The parameter detected is right")
        else:
            print("The parameter of anomaly is wrong")
        plt.plot(df[ anomalies_array[k][0] ])
        plt.axvspan(anomalies_array[k][1], anomalies_array[k][2], color='red', alpha=0.5)
        plt.show()

        print("The diagonal loss : ")
        print("The detected parameter : "+data.params_of_interest[int(loss[k][0,1])])
        plt.plot(loss[k][:,0], loss[k][:,3])
        plt.axvspan(anomalies_array[k][1], anomalies_array[k][2], color='red', alpha=0.5)
        plt.show()
        print("Real anomaly parameter : " + anomalies_array[k][0])
        plt.plot(loss[k][:,0], loss[k][:,6][np.where(np.asarray(data.params_of_interest)==anomalies_array[k][0])[0][0]])
        plt.axvspan(anomalies_array[k][1], anomalies_array[k][2], color='red', alpha=0.5)
        plt.show()
        
        print("The detected parameter : "+data.params_of_interest[int(loss[k][0,1])])        
        print("The diagonal sum loss : ")
        plt.plot(loss[k][:,0], loss[k][:,4])
        plt.axvspan(anomalies_array[k][1], anomalies_array[k][2], color='red', alpha=0.5)
        plt.show()
        print("Real anomaly parameter : " + anomalies_array[k][0])
        plt.plot(loss[k][:,0], loss[k][:,7][np.where(np.asarray(data.params_of_interest)==anomalies_array[k][0])[0][0]])
        plt.axvspan(anomalies_array[k][1], anomalies_array[k][2], color='red', alpha=0.5)
        plt.show()
        print("#####################################")




if name == "__main__":
    
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
    
    loss4, Dia4, Dia_sum4 = max_sum_loss(autoencoder_flight4, matrix4, labels4, 15)
    show_anomalies_basic(loss4, Dia4, Dia_sum4)
