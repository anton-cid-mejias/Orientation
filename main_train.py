from src.data import FiguresDataset, load_figures_data
from src import train, model, evaluate, detect
from src.config import Config

def main():
    train_path = "data/Cube/train"
    val_path = "data/Cube/val"
    config = Config()
    weights = "logs/Cube_test/orientations_7900.h5"

    # Loading dataset
    dataset_train = FiguresDataset()
    dataset_train.load_figures(train_path, "train_annotations.json")
    dataset_train.prepare()

    dataset_val = FiguresDataset()
    dataset_val.load_figures(val_path, "val_annotations.json")
    dataset_val.prepare()

    train_images, train_orientations = load_figures_data(dataset_train, config)
    val_images, val_orientations = load_figures_data(dataset_val, config)

    # Loading model and weights
    or_model = model.OrientationModel("logs", config)
    if weights is not None:
        or_model.compile(weights)

    # Training
    train.train(or_model, (train_images, train_orientations), (val_images, val_orientations),
                epochs=config.EPOCHS)

    # Evaluation
    #evaluate.evaluate(or_model, val_images, val_orientations)

if __name__=="__main__":
    main()