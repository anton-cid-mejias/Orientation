from scipy.spatial.transform import Rotation as R

def train(model, train_data, val_data, epochs=100):

    # Transform euler angles to rotation matrices
    x_train = train_data[0]
    y_train = train_data[1]
    y_train = R.from_euler('ZYX', y_train, degrees=True).as_matrix()

    x_val= val_data[0]
    y_val = val_data[1]
    y_val = R.from_euler('ZYX', y_val, degrees=True).as_matrix()

    # Check NaNs in the training and validation datasets
    #print(np.isnan(x_train).any())
    #print(np.isnan(y_train).any())
    #print(np.isnan(x_val).any())
    #print(np.isnan(y_val).any())

    model.train(x_train, y_train, x_val, y_val, epochs)