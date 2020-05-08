from src.coco_data import FiguresDataset, load_figures_data
from src import model, detect, visualize
from src.config import Config

def main():
    val_path = "data/Cube/val"
    weights = "logs/Cube_test/orientations_7900.h5"
    config = Config()

    dataset_val = FiguresDataset()
    dataset_val.load_figures(val_path, "val_annotations.json")
    dataset_val.prepare()

    val_images, _ = load_figures_data(dataset_val, config)

    # Loading model and weights
    or_model = model.OrientationModel("logs", config)
    if weights is not None:
        or_model.compile(weights)

    # Inference
    predictions = detect.detect(or_model, val_images)
    visualize.show_results(val_images, predictions)

if __name__=="__main__":
    main()