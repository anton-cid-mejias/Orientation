from src import model

def detect(model, images):
    output = model.predict(images)
    return output