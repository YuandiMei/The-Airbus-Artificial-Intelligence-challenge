import numpy as np
import scipy as sc
import os

'''
This file is the compute of signature matrix from the multivariate raw signals
The function signature_matrix takes path, filename, length, w, delta_t as input
and returns series of signature matrices as output
The run_sign_matrix is the executing function of function signature_matrix
'''

#The path of files path1, the list of files dirs1, and the path of validation dataset path_valid, and the list of files of validation dataset dirs_valid. 
path1='Data/asset'
dirs1=os.listdir(path1)
path_valid = 'Data/valid_data'
dirs_valid = os.listdir(path_valid)

# The list of parameter names. params is the list of all parameters except the parameters of interest. params2 is the list of all parameters. params_of_interest is the list of parameters of interest.
params=[]
for i in range(90):
    params.append('parameter_'+str(i))
params
del params[86]
del params[53]
del params[48:51]
del params[9:13]

# The list of all parameters
params2=[]
for i in range(90):
    params2.append('parameter_'+str(i))

# The list of parameters of interest
params_of_interest=['parameter_48','parameter_49','parameter_50','parameter_53','parameter_86','parameter_11','parameter_12','parameter_9','parameter_10']

# The function to transform from initial multivariate time sequences to signature matrices.
def signature_matrix(path, filename, length, w, deltat):
    # The signature matrix function
    # Take one parameter as input, and w as length of window, then output two videos, one for Grounding, one for Flight 
    # The deltat is the number of points per which we pick one point from. We take deltat = 20
    # The length is the number of matrices in each video. We take length = 100 or 20
    # The w is the number points to compute in each matrix. We take w = 15 or 5
    df = pd.read_csv(path+'/'+filename, index_col=0)
    shape = df.shape[0]

    if shape > (length+w-1) * deltat:
        Bool=True
        # 1. Select 1 point per each w(=30) points
        iteration=[]
        shape2 = deltat
        while shape2<shape:
            iteration.append(shape2)
            shape2 = shape2 + deltat
        #iteration = [60, 120, 180, ...]
        
        # 2. Select (length+w-1)(=100+20-1) points per block to construct blocks that can be windowed in length blocks
        step_of_windowing = 1
        init = w+(length-1)*step_of_windowing
        itit = init
        iteration2 = []
        while itit<len(iteration):
            iter3 = []
            for i in range(init):
                iter3.append(iteration[itit-init+i])
            iteration2.append(iter3)
            itit = itit+init-2
            
        # Also 2. Add the last block
        iter3 = []
        #itit = iteration[len(iteration)-1]
        itit = shape-1
        for i in range(init):
            iter3.append( itit-(init-i-1)*deltat )
        iteration2.append(iter3)
        num_of_blocks = len(iteration2)
        
        # 3. Windowing the blocks        
        iteration3 = np.zeros(((num_of_blocks, length, w)))
        for i in range(num_of_blocks):
            for j in range(length):
                for k in range(w):
                    iteration3[i,j,k] = int(iteration2[i][j+k])
         #Shape of iteration3 = (num_of_blocks, length, w)
        final_list=[]        
        if num_of_blocks>1:
            for i in range(num_of_blocks): 
                matrix = np.zeros(((( len(params_of_interest), len(params_of_interest), length, 1 ))))
                for a in range(len(params_of_interest)):
                    for b in range(len(params_of_interest)):
                        for c in range(length):
                            for d in range(1):
                                for e in range(w):
                                    matrix[a,b,c,d] = matrix[a,b,c,d] + (df[params_of_interest[a]][iteration3[i,c,e]]*df[params_of_interest[b]][iteration3[i,c,e]])/w
                final_list.append(matrix)
        else:
            Bool=False
    else:
        Bool = False
        final_list=[]
        iteration3 = np.zeros(((1,1,1)))
    return final_list, Bool, iteration3

def signature_matrixd(path, filename, length, w, deltat):
    # The function of computing the origin and deriative time sequences into signature matrices.
    # Take one parameter as input, and w as length of window, then output two videos, one for Grounding, one for Flight 
    # The deltat is the number of points per which we pick one point from. We take deltat = 20
    # The length is the number of matrices in each video. We take length = 100 or 20
    # The w is the number points to compute in each matrix. We take w = 15 or 5
    dfv = pd.read_csv(path+'/'+filename, index_col=0)
    # path1, dirs1[i]
    shape = dfv.shape[0]
    df = []
    for i in range(len(params_of_interest)):
        df.append(dfv[params_of_interest[i]].as_matrix())
    for i in range(len(params_of_interest)):
        df.append(np.gradient(dfv[params_of_interest[i]]))
    df = np.asarray(df)
    
    if shape > (length+w-1) * deltat:
        Bool=True
        # 1. Select 1 point per each w(=30) points
        iteration=[]
        shape2 = deltat
        while shape2<shape:
            iteration.append(shape2)
            shape2 = shape2 + deltat
        #iteration = [60, 120, 180, ...]
        
    # 2. Select (length+w-1)(=100+20-1) points per block to construct blocks that can be windowed in length blocks
        step_of_windowing = 1
        init = w+(length-1)*step_of_windowing
        itit = init
        iteration2 = []
        while itit<len(iteration):
            iter3 = []
            for i in range(init):
                iter3.append(iteration[itit-init+i])
            iteration2.append(iter3)
            itit = itit+init-2
            
        # Also 2. Add the last block
        iter3 = []
        #itit = iteration[len(iteration)-1]
        itit = shape-1
        for i in range(init):
            iter3.append( itit-(init-i-1)*deltat )
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
                matrix = np.zeros(((( 18, 18, length, 1 ))))
                for a in range(18):
                    for b in range(18):
                        for c in range(length):
                            for d in range(1):
                                for e in range(w):
                                    matrix[a,b,c,d] = matrix[a,b,c,d] + (df[a][int(iteration3[i,c,e])]*df[b][int(iteration3[i,c,e])])/w
                final_list.append(matrix)
        else:
            Bool=False
    else:
        Bool = False
        final_list=[]
        iteration3 = np.zeros(((1,1,1)))
    return final_list, Bool, iteration3
##################################################################
# The main executing function to execute function signature_matrix
def run_sign_matrix(length1, length2, w1, w2, deltat):
    # In this function, we will execute the signature_matrix function and save the computed matrices into desktops "path_mat" and "path_mat_valid", with time labels correspond "path_lab" and "path_mat_valid". The path_mat1 and path_lab1 registrate the signature matrices computed of w=5 and length=100, the path_mat2 and path_lab2 registrate the signature matrices computed of w=15 and length=100, the path_mat3 and path_lab3 registrate the signature matrices computed of w=5 and length=20,  
    #length1=100, length2=20, w1=5, w2=15, deltat=20    
    # The signature matrices of length=100 and w=5 from training data
    path_mat1 = 'Data/signature_matrix_1'
    path_lab1 = 'Data/signature_labels_1'
    #length=100
    #w=5
    #deltat=20
    Total_num_of_blocks = 0
    for i in range(len(dirs1)):
        final_list, Bool, iteration3 = signature_matrix(path1, dirs1[i], length1, w1, deltat)
        if Bool == True:
            print("num of blocks = "+ str(len(final_list)))
            Total_num_of_blocks = Total_num_of_blocks+len(final_list)
            # Save the list of signature matrices into the path_mat1 with same filename of its original time series in format ".npy"
            np.save(path_mat1+'/'+dirs1[i]+'.npy', final_list)
            # Save the list of time labels correspond to the signature matrices into the path_lab1 with same filename of its original time series in format ".npy"
            np.save(path_lab1+'/'+dirs1[i]+'.npy', iteration3)
            print(str(i+1)+"e done!")
        else:
            print(str(i+1)+"e failed!")        
    print("Total num of blocks = "+ str(Total_num_of_blocks))

    # The signature matrices of length=100 and w=5 from validation data
    path_mat_valid1 = 'Data/valid_matrix_1'
    path_lab_valid1 = 'Data/valid_labels_1'
    #length=100
    #w=5
    #deltat=20
    for i in range(len(dirs_valid)):
        final_list, Bool, iteration3 = signature_matrix(path_valid, dirs_valid[i], length1, w1, deltat)
        if Bool == True:
            print("num of blocks = "+ str(len(final_list)))
            Total_num_of_blocks = Total_num_of_blocks+len(final_list)
            np.save(path_mat_valid1+'/'+dirs_valid[i]+'.npy', final_list)
            np.save(path_lab_valid1+'/'+dirs_valid[i]+'.npy', iteration3)
            print(str(i+1)+"e done!")
        else:
            print(str(i+1)+"e failed!")
    print("Total num of blocks = "+ str(Total_num_of_blocks))

    # The signature matrices of length=100 and w=15 from training data
    path_mat2 = 'Data/signature_matrix_2'
    path_lab2 = 'Data/signature_labels_2'
    #length=100
    #w = 15
    #deltat=20
    Total_num_of_blocks = 0
    for i in range(len(dirs1)):
        final_list, Bool, iteration3 = signature_matrix(path1, dirs1[i], length1, w2, deltat)
        if Bool == True:
            print("num of blocks = "+ str(len(final_list)))
            Total_num_of_blocks = Total_num_of_blocks+len(final_list)
            np.save(path_mat2+'/'+dirs1[i]+'.npy', final_list)
            np.save(path_lab2+'/'+dirs1[i]+'.npy', iteration3)
            print(str(i+1)+"e done!")
        else:
            print(str(i+1)+"e failed!")
    print("Total num of blocks = "+ str(Total_num_of_blocks))

    # The signature matrices of length=100 and w=15 from validation data
    path_mat_valid2 = 'Data/valid_matrix_2'
    path_lab_valid2 = 'Data/valid_labels_2'
    length=100
    w = 15
    deltat=20
    Total_num_of_blocks = 0
    for i in range(len(dirs_valid)):
        final_list, Bool, iteration3 = signature_matrix(path_valid, dirs_valid[i], length1, w2, deltat)
        if Bool == True:
            print("num of blocks = "+ str(len(final_list)))
            Total_num_of_blocks = Total_num_of_blocks+len(final_list)
            np.save(path_mat_valid2+'/'+dirs_valid[i]+'.npy', final_list)
            np.save(path_lab_valid2+'/'+dirs_valid[i]+'.npy', iteration3)
            print(str(i+1)+"e done!")
        else:
            print(str(i+1)+"e failed!")        
    print("Total num of blocks = "+ str(Total_num_of_blocks))

    # The signature matrices of length=20 and w=5 from training data
    path_mat3 = 'Data/signature_matrix_3'
    path_lab3 = 'Data/signature_labels_3'
    length=20
    w=5
    deltat=20
    Total_num_of_blocks = 0
    for i in range(len(dirs1)):
        final_list, Bool, iteration3 = signature_matrix(path1, dirs1[i], length2, w1, deltat)
        if Bool == True:
            print("num of blocks = "+ str(len(final_list)))
            Total_num_of_blocks = Total_num_of_blocks+len(final_list)
            np.save(path_mat3+'/'+dirs1[i]+'.npy', final_list)
            np.save(path_lab3+'/'+dirs1[i]+'.npy', iteration3)
            print(str(i+1)+"e done!")
        else:
            print(str(i+1)+"e failed!")
    print("Total num of blocks = "+ str(Total_num_of_blocks))

    # The signature matrices of length=20 and w=5 from validation data
    path_mat_valid3 = 'Data/valid_matrix_3'
    path_lab_valid3 = 'Data/valid_labels_3'
    length=20
    w = 5
    deltat=20
    Total_num_of_blocks = 0
    for i in range(len(dirs_valid)):
        final_list, Bool, iteration3 = signature_matrix(path_valid, dirs_valid[i], length2, w1, deltat)
        if Bool == True:
            print("num of blocks = "+ str(len(final_list)))
            Total_num_of_blocks = Total_num_of_blocks+len(final_list)
            np.save(path_mat_valid3+'/'+dirs_valid[i]+'.npy', final_list)
            np.save(path_lab_valid3+'/'+dirs_valid[i]+'.npy', iteration3)
            print(str(i+1)+"e done!")
        else:
            print(str(i+1)+"e failed!")
    print("Total num of blocks = "+ str(Total_num_of_blocks))

    # The signature matrices of length=20 and w=15 from training data
    path_mat4 = 'Data/signature_matrix_4'
    path_lab4 = 'Data/signature_labels_4'
    length=20
    w = 15
    deltat=20
    Total_num_of_blocks = 0
    for i in range(len(dirs1)):
        final_list, Bool, iteration3 = signature_matrix(path1, dirs1[i], length2, w2, deltat)
        if Bool == True:
            print("num of blocks = "+ str(len(final_list)))
            Total_num_of_blocks = Total_num_of_blocks+len(final_list)
            np.save(path_mat4+'/'+dirs1[i]+'.npy', final_list)
            np.save(path_lab4+'/'+dirs1[i]+'.npy', iteration3)
            print(str(i+1)+"e done!")
        else:
            print(str(i+1)+"e failed!")
    print("Total num of blocks = "+ str(Total_num_of_blocks))

    # The signature matrices of length=20 and w=15 from validation data
    path_mat_valid4 = 'Data/valid_matrix_4'
    path_lab_valid4 = 'Data/valid_labels_4'
    length=20
    w = 15
    deltat=20
    Total_num_of_blocks = 0
    for i in range(len(dirs_valid)):
        final_list, Bool, iteration3 = signature_matrix(path_valid, dirs_valid[i], length2, w2, deltat)
        if Bool == True:
            print("num of blocks = "+ str(len(final_list)))
            Total_num_of_blocks = Total_num_of_blocks+len(final_list)
            np.save(path_mat_valid4+'/'+dirs_valid[i]+'.npy', final_list)
            np.save(path_lab_valid4+'/'+dirs_valid[i]+'.npy', iteration3)
            print(str(i+1)+"e done!")
        else:
            print(str(i+1)+"e failed!")
    print("Total num of blocks = "+ str(Total_num_of_blocks))
    
    return None

if __name__ ="__main__":
    # We run all algorithms in the run_sign_matrix
    run_sign_matrix(length1=100, length2=20, w1=5, w2=15, deltat=20)
