import data
import convlstm_autoencoder as ca

import data
import os
import numpy as np

#This file is the an executing file, which contains loading signature matrices computed in the file data.py, and training the autoencoders of file convlstm_autoencoder.py . After training, it saves the models into files json.
if __name__ == '__main__':

    # There are in total 8 frameworks for the Airbus Commercial Aircraft Challenge. There are two lengths, 100 and 20, two options of w, 15 and 5, and two different dataset, data of ground test and flight test. The 1st flight and ground autoencoder are of length=100 and w=5, the 2nd flight and ground autoencoder are of length=100 and w=15. The 3rd flight and ground autoencoder are of length=20 and w=5, and the 4th flight and ground autoencoder are of length=20 and w=15. Among them all, the 4th flight and ground test works the best on detecting anomalies.

    # Load pre-computed signature matrix from path "path_mat", load list of filenames from path "dirs_mat", load time labels from path "path_lab". 
    #######################################
    path_mat1 = 'Data/signature_matrix_1'
    dirs_mat1 = os.listdir(path_mat1)
    path_lab1 = 'Data/signature_labels_1'
    # The signature matrices are registrated in the list "matrix_flight" and "matrix_ground"
    matrix_flight1 = []
    matrix_ground1 = []
    # The time labels correspond to the signature matrices are registrated in the list "labels_flight" and "labels_ground"
    labels_flight1 = []
    labels_ground1 = []
    # The filenames correspond to the 2 previous lists are registrated in the list "fileic_flight" and "fileic_ground"
    fileic_flight1 = []
    fileic_ground1 = []
    # The "length" is the length to be used in the convolutional LSTM blocks. And the data correspond all have the same length with the framework.
    length=100
    print("Total number of files: "+str(len(dirs_mat1)))
    for i in range(len(dirs_mat1)):
        matrix = np.load(path_mat1+'/'+dirs_mat1[i])
        labels = np.load(path_lab1+'/'+dirs_mat1[i])
        if dirs_mat1[i][0] == 'B':
            for j in range(matrix.shape[0]): 
                matrix_flight1.append(matrix[j,:,:,:,:])
                labels_flight1.append(labels[j,:,:])
                fileic_flight1.append(dirs_mat1[i])
        else:
            for j in range(matrix.shape[0]): 
                matrix_ground1.append(matrix[j,:,:,:,:])
                labels_ground1.append(labels[j,:,:])
                fileic_ground1.append(dirs_mat1[i])

    ########################################
    #Load pre-computed signature matrix from path "path_mat", load list of filenames from path "dirs_mat", load time labels from path "path_lab".
    path_mat2 = 'Data/signature_matrix_2'
    dirs_mat2 = os.listdir(path_mat2)
    path_lab2 = 'Data/signature_labels_2'
    matrix_flight2 = []
    matrix_ground2 = []
    labels_flight2 = []
    labels_ground2 = []
    fileic_flight2 = []
    fileic_ground2 = []
    length=100
    print("Total number of files: "+str(len(dirs_mat2)))
    for i in range(len(dirs_mat2)):
        matrix = np.load(path_mat2+'/'+dirs_mat2[i])
        labels = np.load(path_lab2+'/'+dirs_mat2[i])
        if dirs_mat2[i][0] == 'B':
            for j in range(matrix.shape[0]): 
                matrix_flight2.append(matrix[j,:,:,:,:])
                labels_flight2.append(labels[j,:,:])
                fileic_flight2.append(dirs_mat2[i])
        else:
            for j in range(matrix.shape[0]): 
                matrix_ground2.append(matrix[j,:,:,:,:])
                labels_ground2.append(labels[j,:,:])
                fileic_ground2.append(dirs_mat2[i])


    ########################################
    #Load pre-computed signature matrix from path "path_mat", load list of filenames from path "dirs_mat", load time labels from path "path_lab".
    path_mat3 = 'Data/signature_matrix_3'
    dirs_mat3 = os.listdir(path_mat3)
    path_lab3 = 'Data/signature_labels_3'
    matrix_flight3 = []
    matrix_ground3 = []
    labels_flight3 = []
    labels_ground3 = []
    fileic_flight3 = []
    fileic_ground3 = []
    # The "length" is the length to be used in the convolutional LSTM blocks. And the data correspond all have the same length with the framework. Here, length=20.
    length=20
    print("Total number of files: "+str(len(dirs_mat3)))
    for i in range(len(dirs_mat3)):
        matrix = np.load(path_mat3+'/'+dirs_mat3[i])
        labels = np.load(path_lab3+'/'+dirs_mat3[i])
        if dirs_mat3[i][0] == 'B':
            for j in range(matrix.shape[0]): 
                matrix_flight3.append(matrix[j,:,:,:,:])
                labels_flight3.append(labels[j,:,:])
                fileic_flight3.append(dirs_mat3[i])
        else:
            for j in range(matrix.shape[0]): 
                matrix_ground3.append(matrix[j,:,:,:,:])
                labels_ground3.append(labels[j,:,:])
                fileic_ground3.append(dirs_mat3[i])

    ########################################
    #Load pre-computed signature matrix from path "path_mat", load list of filenames from path "dirs_mat", load time labels from path "path_lab".
    path_mat4 = 'Data/signature_matrix_4'
    dirs_mat4 = os.listdir(path_mat4)
    path_lab4 = 'Data/signature_labels_4'
    matrix_flight4 = []
    matrix_ground4 = []
    labels_flight4 = []
    labels_ground4 = []
    fileic_flight4 = []
    fileic_ground4 = []
    length=20
    print("Total number of files: "+str(len(dirs_mat4)))
    for i in range(len(dirs_mat4)):
        matrix = np.load(path_mat4+'/'+dirs_mat4[i])
        labels = np.load(path_lab4+'/'+dirs_mat4[i])
        if dirs_mat4[i][0] == 'B':
            for j in range(matrix.shape[0]): 
                matrix_flight4.append(matrix[j,:,:,:,:])
                labels_flight4.append(labels[j,:,:])
                fileic_flight4.append(dirs_mat4[i])
        else:
            for j in range(matrix.shape[0]): 
                matrix_ground4.append(matrix[j,:,:,:,:])
                labels_ground4.append(labels[j,:,:])
                fileic_ground4.append(dirs_mat4[i])

    # We transform those lists of matrices into numpy array to become training data for the autoencoder. The nwe print the shape of those numpy arrays.
    ###############################
    X_train_flight1 = np.asarray(matrix_flight1)
    X_train_ground1 = np.asarray(matrix_ground1)
    X_label_flight1 = np.asarray(labels_flight1)
    X_label_ground1 = np.asarray(labels_ground1)
    print("1st flight train data shape = "+str(X_train_flight1.shape))
    print("1st flight train label shape = "+str(X_label_flight1.shape))
    print("1st flight file fileic shape = "+str(len(fileic_flight1)))
    print("1st ground train data shape = "+str(X_train_ground1.shape))
    print("1st ground train label shape = "+str(X_label_ground1.shape))
    print("1st ground file fileic shape = "+str(len(fileic_ground1)))
    ###############################
    X_train_flight2 = np.asarray(matrix_flight2)
    X_train_ground2 = np.asarray(matrix_ground2)
    X_label_flight2 = np.asarray(labels_flight2)
    X_label_ground2 = np.asarray(labels_ground2)
    print("2nd flight train data shape = "+str(X_train_flight2.shape))
    print("2nd flight train label shape = "+str(X_label_flight2.shape))
    print("2nd flight file fileic shape = "+str(len(fileic_flight2)))
    print("2nd ground train data shape = "+str(X_train_ground2.shape))
    print("2nd ground train label shape = "+str(X_label_ground2.shape))
    print("2nd ground file fileic shape = "+str(len(fileic_ground2)))
    ###############################
    X_train_flight3 = np.asarray(matrix_flight3)
    X_train_ground3 = np.asarray(matrix_ground3)
    X_label_flight3 = np.asarray(labels_flight3)
    X_label_ground3 = np.asarray(labels_ground3)
    print("3rd flight train data shape = "+str(X_train_flight3.shape))
    print("3rd flight train label shape = "+str(X_label_flight3.shape))
    print("3rd flight file fileic shape = "+str(len(fileic_flight3)))
    print("3rd ground train data shape = "+str(X_train_ground3.shape))
    print("3rd ground train label shape = "+str(X_label_ground3.shape))
    print("3rd ground file fileic shape = "+str(len(fileic_ground3)))
    ###############################
    X_train_flight4 = np.asarray(matrix_flight4)
    X_train_ground4 = np.asarray(matrix_ground4)
    X_label_flight4 = np.asarray(labels_flight4)
    X_label_ground4 = np.asarray(labels_ground4)
    print("4th flight train data shape = "+str(X_train_flight4.shape))
    print("4th flight train label shape = "+str(X_label_flight4.shape))
    print("4th flight file fileic shape = "+str(len(fileic_flight4)))
    print("4th ground train data shape = "+str(X_train_ground4.shape))
    print("4th ground train label shape = "+str(X_label_ground4.shape))
    print("4th ground file fileic shape = "+str(len(fileic_ground4)))
    ###############################

    #Training phase
    epochs = 20
    batch_size = 20
    # In the training phase, we use the matrices loaded from the previous files, into the frameworks constructed in the file "convlstm_autoencoder.py". 
    ############ 1st training ###################
    print("Training autoencoder_flight1")
    print("###################")
    ca.autoencoder_flight1.fit(X_train_flight1, X_train_flight1, batch_size=batch_size, epochs=epochs, verbose = 1)
    print("Training autoencoder_ground1")
    print("###################")
    ca.autoencoder_ground1.fit(X_train_ground1, X_train_ground1, batch_size=batch_size, epochs=epochs, verbose = 1)
    ############ 2nd training ###################
    print("Training autoencoder_flight2")
    print("###################")
    ca.autoencoder_flight2.fit(X_train_flight2, X_train_flight2, batch_size=batch_size, epochs=epochs, verbose = 1)
    print("Training autoencoder_ground2")
    print("###################")
    ca.autoencoder_ground2.fit(X_train_ground2, X_train_ground2, batch_size=batch_size, epochs=epochs, verbose = 1)
    ############ 3rd training ###################
    print("Training autoencoder_flight3")
    print("###################")
    ca.autoencoder_flight3.fit(X_train_flight3, X_train_flight3, batch_size=batch_size, epochs=epochs, verbose = 1)
    print("Training autoencoder_ground3")
    print("###################")
    ca.autoencoder_ground3.fit(X_train_ground3, X_train_ground3, batch_size=batch_size, epochs=epochs, verbose = 1)
    ############ 4th training ###################
    print("Training autoencoder_flight4")
    print("###################")
    ca.autoencoder_flight4.fit(X_train_flight4, X_train_flight4, batch_size=batch_size, epochs=epochs, verbose = 1)
    print("Training autoencoder_ground4")
    print("###################")
    ca.autoencoder_ground4.fit(X_train_ground4, X_train_ground4, batch_size=batch_size, epochs=epochs, verbose = 1)
    print("Training autoencoder_ground4")
    print("###################")
    ca.autoencoder_ground4.fit(X_train_ground4, X_train_ground4, batch_size=batch_size, epochs=epochs, verbose = 1)
    
    #Save the model as json file.
    ##########################
    path="Models"
    model_json_flight1 = ca.autoencoder_flight1.to_json()
    with open(path+"/model_flight1.json", "w") as json_file:
        json_file.write(model_json_flight1)
    # serialize weights to HDF5
    ca.autoencoder_flight1.save_weights(path+"/model_flight1.h5")
    print("Saved flight model 1 to disk")
    ###########################
    model_json_ground1 = ca.autoencoder_ground1.to_json()
    with open(path+"/model_ground1.json", "w") as json_file:
        json_file.write(model_json_ground1)
    # serialize weights to HDF5
    ca.autoencoder_ground1.save_weights(path+"/model_ground1.h5")
    print("Saved ground model1 to disk")

    ##########################
    model_json_flight2 = ca.autoencoder_flight2.to_json()
    with open(path+"/model_flight2.json", "w") as json_file:
        json_file.write(model_json_flight2)
    # serialize weights to HDF5
    ca.autoencoder_flight2.save_weights(path+"/model_flight2.h5")
    print("Saved flight model 2 to disk")
    ###########################
    model_json_ground2 = ca.autoencoder_ground2.to_json()
    with open(path+"/model_ground2.json", "w") as json_file:
        json_file.write(model_json_ground2)
    # serialize weights to HDF5
    ca.autoencoder_ground2.save_weights(path+"/model_ground2.h5")
    print("Saved ground model 2 to disk")

    ##########################
    model_json_flight3 = ca.autoencoder_flight3.to_json()
    with open(path+"/model_flight3.json", "w") as json_file:
        json_file.write(model_json_flight3)
    # serialize weights to HDF5
    ca.autoencoder_flight3.save_weights(path+"/model_flight3.h5")
    print("Saved flight model 3 to disk")
    ###########################
    model_json_ground3 = ca.autoencoder_ground3.to_json()
    with open(path+"/model_ground3.json", "w") as json_file:
        json_file.write(model_json_ground3)
    # serialize weights to HDF5
    ca.autoencoder_ground3.save_weights(path+"/model_ground3.h5")
    print("Saved ground model 3 to disk")

    ##########################
    model_json_flight4 = ca.autoencoder_flight4.to_json()
    with open(path+"/model_flight4.json", "w") as json_file:
        json_file.write(model_json_flight4)
    # serialize weights to HDF5
    ca.autoencoder_flight4.save_weights(path+"/model_flight4.h5")
    print("Saved flight model 4 to disk")
    ###########################
    model_json_ground4 = ca.autoencoder_ground4.to_json()
    with open(path+"/model_ground4.json", "w") as json_file:
        json_file.write(model_json_ground4)
    # serialize weights to HDF5
    ca.autoencoder_ground4.save_weights(path+"/model_ground4.h5")
    print("Saved ground model 4 to disk")
