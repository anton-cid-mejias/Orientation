import keras.layers as KL
import keras
from keras.regularizers import l2
import math

from src import utils, coco_data, loss

def orientation_graph(input_image):

    x = KL.Conv2D(16, (5, 5), padding="valid", activation='relu', kernel_regularizer=l2(0.0001),
                  bias_regularizer=l2(0.0001), name="or_conv1")(input_image)
    x = KL.MaxPooling2D(pool_size=(2, 2), name="or_pool1")(x)
    x = KL.BatchNormalization(name='or_conv_bn1')(x)

    x = KL.Conv2D(32, (5, 5), padding="valid", activation='relu', kernel_regularizer=l2(0.0001),
                  bias_regularizer=l2(0.0001), name="or_conv2")(x)
    x = KL.MaxPooling2D(pool_size=(2, 2), name="or_pool2")(x)
    x = KL.BatchNormalization(name='or_conv_bn2')(x)

    x = KL.Conv2D(64, (3, 3), padding="valid", activation='relu', kernel_regularizer=l2(0.0001),
                  bias_regularizer=l2(0.0001), name="or_conv3")(x)
    x = KL.MaxPooling2D(pool_size=(2, 2), name="or_pool3")(x)
    x = KL.BatchNormalization(name='or_conv_bn3')(x)

    x = KL.Conv2D(128, (3, 3), padding="valid", activation='relu', kernel_regularizer=l2(0.0001),
                  bias_regularizer=l2(0.0001), name="or_conv4")(x)
    x = KL.MaxPooling2D(pool_size=(2, 2), name="or_pool4")(x)
    x = KL.BatchNormalization(name='or_conv_bn4')(x)

    x = KL.Flatten()(x)

    x = KL.Dense(512, name="or_dense1", kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001),)(x)
    x = KL.LeakyReLU(alpha=0.3)(x)
    x = KL.BatchNormalization(name='or_bn1')(x)

    x = KL.Dense(512, name="or_dense2", kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001),)(x)
    x = KL.LeakyReLU(alpha=0.3)(x)
    x = KL.BatchNormalization(name='or_bn2')(x)

    x = KL.Dense(512, name="or_dense3", kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001),)(x)
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

    def compile(self, weights=None):
        optimizer = keras.optimizers.Adam(lr=self.config.LEARNING_RATE)

        if weights is not None:
            print("Loading weights \"%s\"..." % weights)
            self.keras_model.load_weights(weights)

        if self.config.LOSS == "euclidean":
            self.keras_model.compile(optimizer=optimizer, loss=loss.euc_dist_keras)
        elif self.config.LOSS == "geodesic":
            self.keras_model.compile(optimizer=optimizer, loss=loss.orientation_loss)
        else:
            raise Exception("Incorrect loss name in the configuration file")

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
        tensorboard_cbk = keras.callbacks.TensorBoard(log_dir='logs',
                                                      histogram_freq=0,
                                                      batch_size=self.config.BATCH_SIZE)
        callbacks.append(tensorboard_cbk)

        # Scheduler
        def scheduler(epoch):
            if epoch < 1800:
                return self.config.LEARNING_RATE
            elif epoch < 2400:
                return self.config.LEARNING_RATE * 0.1
            else:
                return self.config.LEARNING_RATE * 0.01

        scheduler_cbk = keras.callbacks.LearningRateScheduler(scheduler)
        callbacks.append(scheduler_cbk)

        csv_logger = keras.callbacks.CSVLogger("logs/model_history_log.csv", append=True)
        callbacks.append(csv_logger)

        # Data generator
        steps = math.ceil(images.shape[0] / self.config.BATCH_SIZE)
        train_generator = coco_data.data_generator(images, gt_orientations,
                                                   batch_size=self.config.BATCH_SIZE,
                                                   augmentation=True)

        '''
        history = self.keras_model.fit(
            images, gt_orientations,
            batch_size=self.config.BATCH_SIZE,
            epochs=epochs,
            validation_data=(val_images, val_gt_orientations),
            callbacks=callbacks
        )'''

        history = self.keras_model.fit_generator(
            train_generator,
            steps_per_epoch=steps,
            epochs=epochs,
            validation_data= (val_images, val_gt_orientations),
            callbacks=callbacks,
            initial_epoch=self.config.INITIAL_EPOCH
        )
        print('\nhistory dict:', history.history)

    def evaluate(self, x_test, y_test):
        results = self.keras_model.evaluate(x_test, y_test)
        return results

    def predict(self, image):
        output = self.keras_model.predict(image)
        return output