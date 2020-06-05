class Config(object):

    LEARNING_RATE = 0.00001
    INPUT_SIZE = 128
    BATCH_SIZE = 25
    EPOCHS = 5000
    AUGMENTATION = True
    LOSS = "euclidean" #geodesic or euclidean
    INITIAL_EPOCH = 0