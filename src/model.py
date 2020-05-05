import keras.layers as KL
import keras
import tensorflow as tf
from keras.regularizers import l2

from src import utils, loss

def orientation_graph(input_image):

    x = KL.Conv2D(16, (5, 5), padding="valid", activation='relu', kernel_regularizer=l2(0.01),
                  bias_regularizer=l2(0.01), name="or_conv1")(input_image)
    x = KL.MaxPooling2D(pool_size=(2, 2), name="or_pool1")(x)
    x = KL.BatchNormalization(name='or_conv_bn1')(x)

    x = KL.Conv2D(32, (5, 5), padding="valid", activation='relu', kernel_regularizer=l2(0.01),
                  bias_regularizer=l2(0.01), name="or_conv2")(x)
    x = KL.MaxPooling2D(pool_size=(2, 2), name="or_pool2")(x)
    x = KL.BatchNormalization(name='or_conv_bn2')(x)

    x = KL.Conv2D(64, (3, 3), padding="valid", activation='relu', kernel_regularizer=l2(0.01),
                  bias_regularizer=l2(0.01), name="or_conv3")(x)
    x = KL.MaxPooling2D(pool_size=(2, 2), name="or_pool3")(x)
    x = KL.BatchNormalization(name='or_conv_bn3')(x)

    x = KL.Conv2D(128, (3, 3), padding="valid", activation='relu', kernel_regularizer=l2(0.01),
                  bias_regularizer=l2(0.01), name="or_conv4")(x)
    x = KL.MaxPooling2D(pool_size=(2, 2), name="or_pool4")(x)
    x = KL.BatchNormalization(name='or_conv_bn4')(x)

    x = KL.Flatten()(x)

    x = KL.Dense(1024, name="or_dense1", kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01),)(x)
    x = KL.LeakyReLU(alpha=0.3)(x)
    x = KL.BatchNormalization(name='or_bn1')(x)

    x = KL.Dense(1024, name="or_dense2", kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01),)(x)
    x = KL.LeakyReLU(alpha=0.3)(x)
    x = KL.BatchNormalization(name='or_bn2')(x)

    x = KL.Dense(1024, name="or_dense3", kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01),)(x)
    x = KL.LeakyReLU(alpha=0.3)(x)
    x = KL.BatchNormalization(name='or_bn3')(x)

    x = KL.Dense(6, name='6d_output')(x)

    r_matrices = KL.Lambda(lambda t: utils.compute_rotation_matrix_from_ortho6d(t))(x)

    #angles = KL.Lambda(lambda  x: utils.compute_euler_angles_from_rotation_matrices(x))(r_matrices)

    return r_matrices#, angles

class OrientationModel():

    def __init__(self, model_dir, config):
        """
        model_dir: Directory to save training logs and trained weights
        """
        self.model_dir = model_dir
        self.config = config
        self.keras_model = self.build()

    def build(self):

        inputs = []
        input_image = KL.Input(shape=(self.config.INPUT_SIZE, self.config.INPUT_SIZE, 3),
                          name="image_input")
        inputs.append(input_image)

        r_matrices = orientation_graph(input_image)
        model = keras.Model(inputs=inputs, outputs=r_matrices, name="orientation_model")
        model.summary()

        return model

    def compile(self):
        optimizer = keras.optimizers.Adam(lr=self.config.LEARNING_RATE)

        self.keras_model.compile(optimizer=optimizer, loss=loss.orientation_loss_graph)

    def train(self, images, gt_orientations, val_images, val_gt_orientations, epochs):

        # Checkpoint saver
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath='logs/orientations_{epoch}.h5',
                # Path where to save the model
                # The two parameters below mean that we will overwrite
                # the current checkpoint if and only if
                # the `val_loss` score has improved.
                save_weights_only=True,
                save_best_only=True,
                monitor='val_loss',
                verbose=1,
                period=100)
        ]

        # Tensorboard
        tensorboard_cbk = keras.callbacks.TensorBoard(log_dir='logs')
        callbacks.append(tensorboard_cbk)

        # Scheduler
        def scheduler(epoch):
            if epoch < 10000:
                return self.config.LEARNING_RATE
            else:
                return self.config.LEARNING_RATE * 0.1

        scheduler_cbk = keras.callbacks.LearningRateScheduler(scheduler)
        callbacks.append(scheduler_cbk)

        history = self.keras_model.fit(
            images, gt_orientations,
            batch_size=self.config.BATCH_SIZE,
            epochs=epochs,
            validation_data= (val_images, val_gt_orientations),
            callbacks=callbacks
        )
        print('\nhistory dict:', history.history)