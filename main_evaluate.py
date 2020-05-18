from src.coco_data import FiguresDataset, load_figures_data
from src import model, detect, visualize, orientation_data, utils, coco_data
from src.config import Config
from scipy.spatial.transform import Rotation as R

def main_coco():
    val_path = "data/Octahedron/val"
    weights = "logs/Octahedron_aug/orientations_4900.h5"
    evaluation_dir = "evaluation"
    config = Config()

    dataset_val = FiguresDataset()
    dataset_val.load_figures(val_path, "val_annotations.json")
    dataset_val.prepare()

    val_images, val_orientations = load_figures_data(dataset_val, config)

    # Loading model and weights
    or_model = model.OrientationModel("logs", config)

    or_model.compile(weights)

    # Inference
    predictions = detect.detect(or_model, val_images)

    gt_orientations = R.from_euler('ZYX', val_orientations, degrees=True).as_matrix()
    utils.evaluate(gt_orientations, predictions, dataset_val, evaluation_dir)

    coco_data.save_pred_annotations(predictions, dataset_val, val_path, evaluation_dir)

    visualize.show_results(val_images, predictions, evaluation_dir)

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