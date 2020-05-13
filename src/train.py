from scipy.spatial.transform import Rotation as R
import numpy as np

def train(model, train_data, val_data, epochs=100, angle='euler'):

    # Transform euler angles to rotation matrices
    x_train = train_data[0]
    y_train = train_data[1]
    if angle == 'euler':
        y_train = R.from_euler('ZYX', y_train, degrees=True).as_matrix()
    elif angle == 'quat':
        y_train = R.from_quat(y_train).as_matrix()

    x_val= val_data[0]
    y_val = val_data[1]
    if angle == 'euler':
        y_val = R.from_euler('ZYX', y_val, degrees=True).as_matrix()
    elif angle == 'quat':
        y_val = R.from_quat(y_val).as_matrix()

    # Check NaNs in the training and validation datasets

    if np.isnan(x_train).any() or np.isnan(y_train).any() or np.isnan(x_val).any() or  np.isnan(y_val).any():
        print("The dataset contains NaNs, aborting execution...")
        return

    model.train(x_train, y_train, x_val, y_val, epochs)