def split_data(x, y, percent=0.1):
    cut = int(percent*x.shape[0])
    X_train = x[:-cut]
    Y_train = y[:-cut]
    X_valid = x[-cut:]
    Y_valid = y[-cut:]
    return X_train, Y_train, X_valid, Y_valid