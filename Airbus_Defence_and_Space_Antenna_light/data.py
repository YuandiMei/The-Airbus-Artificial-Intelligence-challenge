import numpy as np
import scipy as sc
import pandas as pd
import zipfile
import math
import utils
import matplotlib.pyplot as plt
import os
'''
This file is the compute of signature matrix from the multivariate raw signals
The function signature_matrix takes path, filename, length, w, delta_t as input
and returns series of signature matrices as output
The run_sign_matrix is the executing function of function signature_matrix
'''

# df2 is the training dataset, df3 is the test dataset, and df4 is the validation dataset. To guarantee the quality of model, we have to delete all data that contains Nan is the training data. But to ensure all anomalies of test and validation data, we keep all data that contains Nan in the test and validation data, and consider Nan as zero.
df2 = pd.read_hdf(path + '/dftrain.h5')
df2 = df2.dropna()
df3 = pd.read_hdf(path + '/dftest.h5')
df4 = pd.read_hdf(path + '/dfvalid.h5')

# The list of ids of training data, test data and validation data are id_list, id_list_test, id_list_valid. And the "parameter_list" is the list of all parameters names.
id_list = list(set(df2["id"]))
id_list_test  = list(set(df3["id"]))
id_list_valid = list(set(df4["id"]))
parameter_list = []
for i in range(35):
    parameter_list.append("p"+str(i+1))
len(parameter_list)
# The "anomaly_list" is the list of known anomalies provided by the challenge organizer. 
anomaly_list = np.loadtxt(path+'/gt_test.csv', skiprows=1, delimiter=',', dtype=int)
#The path of files path
path = 'Data'
def select_matrix(df, id_list):
    big_group = []
    for i in range(len(id_list)): 
        choosed_frame = df.loc[df['id'] == id_list[i]][parameter_list]
        group=choosed_frame.as_matrix()
        big_group.append(group)
    return big_group

def signature_matrix(df, length, w):
    # Take one parameter as input, and w as length of window, then output two videos, one for Grounding, one for Flight 
    # The deltat is the number of points per which we pick one point from. We take deltat = 20
    # The length is the number of matrices in each video. We take length = 60 or 20
    # The w is the number points to compute in each matrix. We take w = 5 or 15
    shape = df.shape[0]
    if shape > length+w-1:
        Bool=True
        # 1. Select (length+w-1)(=100+20-1) points per block to construct blocks that can be windowed in length blocks
        step_of_windowing = 1
        init = w+(length-1)*step_of_windowing
        itit = init
        iteration2 = []
        while itit<shape:
            iter3 = []
            for i in range(init):
                iter3.append(itit-init+i)
            iteration2.append(iter3)
            itit = itit+init-2
        # 2. Add the last block
        iter3 = []
        itit = shape-1
        for i in range(init):
            iter3.append( itit-(init-i-1) )
        iteration2.append(iter3)
        num_of_blocks = len(iteration2)
        # 3. Windowing the blocks        
        iteration3 = np.zeros(((num_of_blocks, length, w)))
        for i in range(num_of_blocks):
            for j in range(length):
                for k in range(w):
                    iteration3[i,j,k] = int(iteration2[i][j+k])
        final_list=[]        
        if num_of_blocks>1:
            for i in range(num_of_blocks): 
                matrix = np.zeros(((( len(parameter_list), len(parameter_list), length, 1 ))))
                for a in range(len(parameter_list)):
                    for b in range(len(parameter_list)):
                        for c in range(length):
                            for d in range(1):
                                w2 = 0
                                for e in range(w):
                                    if not(math.isnan( df[int(iteration3[i,c,e]), a] )):
                                        if not(math.isnan(df[int(iteration3[i,c,e]), b])):
                                            matrix[a,b,c,d] = matrix[a,b,c,d] + (df[int(iteration3[i,c,e]),a]*df[int(iteration3[i,c,e]),b])
                                            w2 = w2+1
                                if w2!=0:
                                    matrix[a,b,c,d] = matrix[a,b,c,d]/w2
            final_list.append(matrix)
        else:
            Bool=False
    else:
        Bool = False
        final_list=[]
        iteration3 = np.zeros(((1,1,1)))
    return final_list, Bool, iteration3



if __name__ == "__main__":
    # We have chosen here two length and two w, which length=20 or 60 and which w=5 or 15. The length is useful for the convolutional LSTM, and the w is for the compute of signature matrices. But in the real experiments, we have only used signature matrices of w=5.
    length_1=20
    w_1=5
    length_2=60
    w_2=15
    matrix3 = []
    # Computing the signature matrices from test dataset.
    test_mat1 = path + '/validation_matrix1'
    test_lab1 = path + '/validation_labels1'
    #First test matrix
    for i in range(len(matrix_test)):
        final_list, Bool, iteration3 = signature_matrix(matrix_test[i], length_1, w_1)
        if Bool == True:
            np.save(test_mat1+'/'+str(id_list_test[i])+'.npy', final_list)
            np.save(test_lab1+'/'+str(id_list_test[i])+'.npy', iteration3)
            print(str(i+1)+"e done!")

    #Second test matrix
    test_mat2 = path + '/validation_matrix2'
    test_lab2 = path + '/validation_labels2'        
    for i in range(len(matrix_test)):
        final_list, Bool, iteration3 = signature_matrix(matrix_test[i], length_2, w_2)
        if Bool == True:
            np.save(test_mat2+'/'+str(id_list_test[i])+'.npy', final_list)
            np.save(test_lab2+'/'+str(id_list_test[i])+'.npy', iteration3)
            print(str(i+1)+"e done!")

    # Computing the signature matrices from validation dataset.
    #First and second valid matrix
    real_valid1 = path + '/real_validation1'
    real_label1 = path + '/real_validlabel1'
    real_valid2 = path + '/real_validation2'
    real_label2 = path + '/real_validlabel2'

    for i in range(len(matrix_valid)):
        final_list, Bool, iteration3 = signature_matrix(matrix_valid[i], length_1, w_1)
        if Bool == True:
            print("num of blocks = "+ str(len(final_list)))
            np.save(real_valid1+'/'+str(id_list_valid[i])+'.npy', final_list)
            np.save(real_label1+'/'+str(id_list_valid[i])+'.npy', iteration3)
            print(str(i+1)+"e done!")
        
    for i in range(len(matrix_valid)):
        final_list, Bool, iteration3 = signature_matrix(matrix_valid[i], length_2, w_2)
        if Bool == True:
            print("num of blocks = "+ str(len(final_list)))
            np.save(real_valid2+'/'+str(id_list_valid[i])+'.npy', final_list)
            np.save(real_label2+'/'+str(id_list_valid[i])+'.npy', iteration3)
            print(str(i+1)+"e done!")

    # Computing the signature matrices from training dataset.
    path_mat1 = path + '/signature_matrix1'
    path_lab1 = path + '/signature_labels1'
    path_mat2 = path + '/signature_matrix2'
    path_lab2 = path + '/signature_labels2'

    length_1=20
    w_1=5
    length_2=60
    w_2=15
    #First training matrix
    for i in range(len(matrix_train)):
        final_list, Bool, iteration3 = signature_matrix(matrix_train[i], length_1, w_1)
        if Bool == True:
            np.save(path_mat1+'/'+str(id_list[i])+'.npy', final_list)
            np.save(path_lab1+'/'+str(id_list[i])+'.npy', iteration3)
            print(str(i+1)+"e done!")

    length=60
    w=15
    Total_num_of_blocks = 0
    for i in range(len(matrix_train)):
        final_list, Bool, iteration3 = signature_matrix(matrix_train[i], length_2, w_2)
        if Bool == True:
            np.save(path_mat2+'/'+str(id_list[i])+'.npy', final_list)
            np.save(path_lab2+'/'+str(id_list[i])+'.npy', iteration3)
            print(str(i+1)+"e done!")

    # Compute matrix in the list
    matrix3 = []
    path_mat3 = path + '/signature_matrix3'
    path_lab3 = path + '/signature_labels3'

    test_mat3 = path + '/validation_matrix3'
    test_lab3 = path + '/validation_labels3'

    length=60
    w=5
    Total_num_of_blocks = 0
    for i in range(len(matrix_train)):
        final_list, Bool, iteration3 = signature_matrix(matrix_train[i], length, w)
        if Bool == True:
            print("num of blocks = "+ str(len(final_list)))
            Total_num_of_blocks = Total_num_of_blocks+len(final_list)
            for j in range(len(final_list)):
                matrix = final_list[j]
                matrix3.append(matrix)
            np.save(path_mat2+'/'+str(id_list[i])+'.npy', final_list)
            np.save(path_lab2+'/'+str(id_list[i])+'.npy', iteration3)

            print(str(i+1)+"e done!")
        else:
            print(str(i+1)+"e failed!")
            
    # Print how many times of loading have been done through "Total_num_of_blocks"
    Total_num_of_blocks = 0
    for i in range(len(matrix_test)):
        final_list, Bool, iteration3 = signature_matrix(matrix_test[i], length, w)
        if Bool == True:
            print("num of blocks = "+ str(len(final_list)))
            Total_num_of_blocks = Total_num_of_blocks+len(final_list)
            for j in range(len(final_list)):
                matrix = final_list[j]
                matrix3.append(matrix)
            np.save(test_mat3+'/'+str(id_list[i])+'.npy', final_list)
            np.save(test_lab3+'/'+str(id_list[i])+'.npy', iteration3)

            print(str(i+1)+"e done!")
        else:
            print(str(i+1)+"e failed!")
    print("Total num of blocks = "+ str(Total_num_of_blocks))

    # Compute matrix in the list
    matrix3 = []
    path_mat4 = path + '/signature_matrix4'
    path_lab4 = path + '/signature_labels4'

    test_mat4 = path + '/validation_matrix4'
    test_lab4 = path + '/validation_labels4'

    length=100
    w=15
    Total_num_of_blocks = 0
    for i in range(len(matrix_train)):
        final_list, Bool, iteration3 = signature_matrix(matrix_train[i], length, w)
        if Bool == True:
            print("num of blocks = "+ str(len(final_list)))
            Total_num_of_blocks = Total_num_of_blocks+len(final_list)
            for j in range(len(final_list)):
                matrix = final_list[j]
                matrix3.append(matrix)
            np.save(path_mat4+'/'+str(id_list[i])+'.npy', final_list)
            np.save(path_lab4+'/'+str(id_list[i])+'.npy', iteration3)
            print(str(i+1)+"e done!")
        else:
            print(str(i+1)+"e failed!")
            
    Total_num_of_blocks = 0            
    for i in range(len(matrix_test)):
        final_list, Bool, iteration3 = signature_matrix(matrix_test[i], length, w)
        if Bool == True:
            print("num of blocks = "+ str(len(final_list)))
            Total_num_of_blocks = Total_num_of_blocks+len(final_list)
            for j in range(len(final_list)):
                matrix = final_list[j]
                matrix3.append(matrix)
            np.save(test_mat4+'/'+str(id_list[i])+'.npy', final_list)
            np.save(test_lab4+'/'+str(id_list[i])+'.npy', iteration3)
            print(str(i+1)+"e done!")
        else:
            print(str(i+1)+"e failed!")
    print("Total num of blocks = "+ str(Total_num_of_blocks))

    real_valid1 = path + '/real_validation1'
    real_label1 = path + '/real_validlabel1'
    length_1=20
    w_1=5

    real_valid2 = path + '/real_validation2'
    real_label2 = path + '/real_validlabel2'
    length_2=60
    w_2=15

    real_valid3 = path + '/real_validation3'
    real_label3 = path + '/real_validlabel3'
    length_3=60
    w_3=5

    real_valid4 = path + '/real_validation4'
    real_label4 = path + '/real_validlabel4'
    length_4=100
    w_4=15

    Total_num_of_blocks = 0
    for i in range(len(matrix_valid)):
        final_list, Bool, iteration3 = signature_matrix(matrix_valid[i], length_1, w_1)
        if Bool == True:
            print("num of blocks = "+ str(len(final_list)))
            Total_num_of_blocks = Total_num_of_blocks+len(final_list)
            np.save(real_valid1+'/'+str(id_list[i])+'.npy', final_list)
            np.save(real_label1+'/'+str(id_list[i])+'.npy', iteration3)
            print(str(i+1)+"e done!")
        else:
            print(str(i+1)+"e failed!")
    print("Total num of blocks = "+ str(Total_num_of_blocks))
            
    Total_num_of_blocks = 0
    for i in range(len(matrix_valid)):
        final_list, Bool, iteration3 = signature_matrix(matrix_valid[i], length_2, w_2)
        if Bool == True:
            print("num of blocks = "+ str(len(final_list)))
            Total_num_of_blocks = Total_num_of_blocks+len(final_list)
            np.save(real_valid2+'/'+str(id_list[i])+'.npy', final_list)
            np.save(real_label2+'/'+str(id_list[i])+'.npy', iteration3)
            print(str(i+1)+"e done!")
        else:
            print(str(i+1)+"e failed!")
    print("Total num of blocks = "+ str(Total_num_of_blocks))
            
    Total_num_of_blocks = 0
    for i in range(len(matrix_valid)):
        final_list, Bool, iteration3 = signature_matrix(matrix_valid[i], length_3, w_1)
        if Bool == True:
            print("num of blocks = "+ str(len(final_list)))
            Total_num_of_blocks = Total_num_of_blocks+len(final_list)
            np.save(real_valid3+'/'+str(id_list[i])+'.npy', final_list)
            np.save(real_label3+'/'+str(id_list[i])+'.npy', iteration3)
            print(str(i+1)+"e done!")
        else:
            print(str(i+1)+"e failed!")
    print("Total num of blocks = "+ str(Total_num_of_blocks))
            
    Total_num_of_blocks = 0
    for i in range(len(matrix_valid)):
        final_list, Bool, iteration3 = signature_matrix(matrix_valid[i], length_4, w_2)
        if Bool == True:
            print("num of blocks = "+ str(len(final_list)))
            Total_num_of_blocks = Total_num_of_blocks+len(final_list)
            np.save(real_valid4+'/'+str(id_list[i])+'.npy', final_list)
            np.save(real_label4+'/'+str(id_list[i])+'.npy', iteration3)
            print(str(i+1)+"e done!")
        else:
            print(str(i+1)+"e failed!")
    print("Total num of blocks = "+ str(Total_num_of_blocks))
            
    Total_num_of_blocks = 0
    for i in range(len(matrix_valid)):
        final_list, Bool, iteration3 = signature_matrix(matrix_valid[i], length_4, w_2)
        if Bool == True3:
            print("num of blocks = "+ str(len(final_list)))
            Total_num_of_blocks = Total_num_of_blocks+len(final_list)
            np.save(real_valid4+'/'+str(id_list[i])+'.npy', final_list)
            np.save(real_label4+'/'+str(id_list[i])+'.npy', iteration3)
            print(str(i+1)+"e done!")
        else:
            print(str(i+1)+"e failed!")
    print("Total num of blocks = "+ str(Total_num_of_blocks))
