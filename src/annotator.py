import json
from datetime import date

class AnnotationsGenerator:
    def __init__(self, description):
        self.licenses = []
        self.images = []
        self.annotations = []
        self.categories = []
        self.categories_dict = {}
        self.info = {
            "description": description,
            "url": "",
            "version": "1.0",
            "year": int(date.today().strftime("%Y")),
            "contributor": "Anton",
            "date_created": date.today().strftime("%Y/%m/%d")
        }
        self.root = {
            "info": self.info,
            "licenses": {},
            "images": self.images,
            "categories": self.categories,
            "annotations": self.annotations,
        }

    def add_image(self, image_name, height, width):
        new_id = len(self.images) + 1
        img_dict = {
            "file_name": image_name,
            "height": height,
            "width": width,
            "id": new_id
        }
        self.images.append(img_dict)

        return new_id

    def get_image_id(self, image_name):
        for image in self.images:
            if image['file_name'] == image_name:
                return image['id']

        return None

    def add_categories(self, categories):
        new_id = len(self.categories) + 1
        for category in categories:
            cat = {
                "supercategory": "figure",
                "id": new_id,
                "name": category
            }
            self.categories.append(cat)
            self.categories_dict[category] = new_id
            new_id += 1
        return

    def get_category_id(self, category):
        return self.categories_dict[category]

    def add_annotation(self, image_id, category, bbox, area, segmentation, orientation):
        new_id = len(self.annotations) + 1
        category_id = self.categories_dict[category]
        if category_id == None:
            raise Exception("Category of annotation not matching the categories of the annotator")
        annotation_dict = {
            "image_id": image_id,
            "category_id": category_id,
            "bbox": bbox, #[top left x position, top left y position, width, height]
            "area": area,
            "segmentation": segmentation,
            "orientation": orientation,  # [anglex, angley, anglez]
            "iscrowd": 0,
            "id": new_id
        }
        self.annotations.append(annotation_dict)
        return

    def get_annotations(self, image_id):
        annotations = []
        for annotation in self.annotations:
            if annotation['image_id'] == image_id:
                annotations.append(annotation)
        return annotations

    def save_json(self, filename, pretty=False):
        with open(filename, 'w') as fp:
            if pretty:
                json.dump(self.root, fp, indent=2)
            else:
                json.dump(self.root, fp)
        return