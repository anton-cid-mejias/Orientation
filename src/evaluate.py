from scipy.spatial.transform import Rotation as R

def evaluate(model, x_test, y_test):
    y_test = R.from_euler('ZYX', y_test, degrees=True).as_matrix()

    results = model.evaluate(x_test, y_test)
    print('Test loss:', results)