from src.data import FiguresDataset, load_figures_data
from src import train, config

def main():
    train_path = "data/Cube/train"
    val_path = "data/Cube/val"
    config_ = config.Config
    weights = None#"logs/orientations_2100.h5"

    dataset_train = FiguresDataset()
    dataset_train.load_figures(train_path, "train_annotations.json")
    dataset_train.prepare()

    dataset_val = FiguresDataset()
    dataset_val.load_figures(val_path, "val_annotations.json")
    dataset_val.prepare()

    train_images, train_orientations = load_figures_data(dataset_train, config_)
    val_images, val_orientations = load_figures_data(dataset_val, config_)

    train.train((train_images, train_orientations), (val_images, val_orientations),
                config_, epochs=config_.EPOCHS, weights=weights)

if __name__=="__main__":
    main()