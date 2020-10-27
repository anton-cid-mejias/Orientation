from src.coco_data import FiguresDataset, load_figures_data
from src import model, detect, visualize, orientation_data, utils, coco_data
from src.config import Config
import time

def main_coco():
    #dataset_path = "data/Detections/Cube_0.9"
    dataset_path = "data/Detections/Final/Cube"
    #dataset_path = "data/Detections/Final/Octahedron"
    #dataset_path = "data/Detections/Final/Needle"
    #dataset_path = "data/Detections/Final/Hexagon_lat"

    #weights = "logs/Final/Cube/orientations_2900.h5"
    #weights = "logs/Final/Cube/orientations_2900.h5"
    #weights = "logs/Final/Needle/orientations_3000.h5"
    #weights = "logs/Final/Octahedron/orientations_2900.h5"
    weights = "logs/Random/Cube/orientations_1200.h5"
    annotations = "prediction_annotations_90.json"
    pred_dir = "predictions/Final"

    config = Config()

    dataset = FiguresDataset()
    dataset.load_figures(dataset_path, annotations)
    dataset.prepare()

    images, orientations, masks = load_figures_data(dataset, config, mask=True)
    #images = utils.apply_mask(images, masks, extra=3)

    # Loading model and weights
    or_model = model.OrientationModel("logs", config)

    or_model.compile(weights)

    # Inference
    start = time.time()
    predictions = detect.detect(or_model, images)
    end = time.time()
    print(end - start)

    coco_data.save_pred_annotations(predictions, dataset, dataset_path, pred_dir)

    visualize.show_results(images, predictions, pred_dir)

#IMPORTANT: IMAGES VALUES IN RANGE [0,1]
def main_or():
    IMAGES_DIR = "data/Cube_2.0/Images"
    FILE_PATH = "data/Cube_2.0/cube_quat_angles.csv"
    config = Config()
    weights = "logs/orientations_2300.h5"

    train_images, train_angles, val_images, val_angles = \
        orientation_data.load_dataset(FILE_PATH, IMAGES_DIR, (128, 128), train_per=0.5)

    # Loading model and weights
    or_model = model.OrientationModel("logs", config)
    if weights is not None:
        or_model.compile(weights)

    # Inference
    predictions = detect.detect(or_model, val_images)
    visualize.show_results(val_images, predictions)

if __name__=="__main__":
    main_coco()