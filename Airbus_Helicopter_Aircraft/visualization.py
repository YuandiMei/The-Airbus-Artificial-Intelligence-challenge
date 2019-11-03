import train
if __name__=="__main__":
    # Visualize all anomalies of the test data. if you need to visualize all normal data of test data as well, then set ""print_normal=True"
    anomalies_array2, sousmission2, len_anom2 = train.mu_sigma_anomalies(train.X_train_wpe2, train.X_test_wpe2, train.k2, train.List, print_normal=True, plot = True)

