import keras.backend as K
import numpy as np
from src import utils
import tensorflow as tf

def deg2rad(angles):
    return angles * np.pi / 180

def orientation_loss(target_matrices, pred_matrices):
    """Loss for orientation regression.
    """
    # Remove batch dimension for simplicity
    #target_orientations = K.reshape(target_orientations, (-1, 3))
    #pred_matrices = K.reshape(pred_matrices, (-1, 3, 3))

    #target_orientations = deg2rad(target_orientations)
    #target_matrices = utils.from_euler(target_orientations)

    thetas = utils.compute_geodesic_distance_from_two_matrices(target_matrices, pred_matrices)
    #thetas = tf.Print(thetas, [thetas])
    loss = tf.math.reduce_mean(thetas)#K.mean(thetas)
    #loss = tf.Print(loss, [loss])
    return loss