from src import model

def train(train_data, val_data, config, epochs=100, weights=None):

    or_model = model.OrientationModel("logs", config)
    or_model.compile(weights)
    or_model.train(train_data[0], train_data[1], val_data[0], val_data[1], epochs)