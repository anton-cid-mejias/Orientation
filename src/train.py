from src import model
from scipy.spatial.transform import Rotation as R
import numpy as np

def train(train_data, val_data, config, epochs=100, weights=None):

    # Transform euler angles to rotation matrices
    x_train = train_data[0]
    y_train = train_data[1]
    y_train = R.from_euler('ZYX', y_train, degrees=True).as_matrix()

    x_val= val_data[0]
    y_val = val_data[1]
    y_val = R.from_euler('ZYX', y_val, degrees=True).as_matrix()

    #print(np.isnan(x_train).any())
    #print(np.isnan(y_train).any())
    #print(np.isnan(x_val).any())
    #print(np.isnan(y_val).any())

    or_model = model.OrientationModel("logs", config)
    or_model.compile(weights)
    or_model.train(x_train, y_train, x_val, y_val, epochs)