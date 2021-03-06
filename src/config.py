class Config(object):

    LEARNING_RATE = 0.0001
    INPUT_SIZE = 128
    BATCH_SIZE = 25
    EPOCHS = 3000
    AUGMENTATION = True
    LOSS = "euclidean" #geodesic or euclidean
    INITIAL_EPOCH = 0