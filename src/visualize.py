import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import os

def show_results(images, predictions, save_dir):
    for i in range(0, images.shape[0]):
        image = images[i, :, :, :]
        rot_matrix = predictions[i, :, :]

        rotation = R.from_matrix(rot_matrix)
        quat = rotation.as_quat()
        euler = rotation.as_euler('ZYX', degrees=True)

        fig, ax = plt.subplots(figsize=(30, 30))

        ax.imshow(image)

        quat_text = "Quaternations: (%.3f, %.3f, %.3f, %.3f)" % (quat[0], quat[1], quat[2], quat[3])
        euler_text = "Euler: x: %.2f, y: %.2f, z: %.2f" % (euler[0], euler[1], euler[2])
        ax.text(10, 10, quat_text, color='black', size=50, bbox=dict(facecolor='w', alpha=.9))
        ax.text(10, 30, euler_text, color='black', size=50, bbox=dict(facecolor='w', alpha=.9))

        print("Saving %s/%i_pred.png..." % (save_dir, i))
        plt.savefig(os.path.join(save_dir, "%i_pred.png" % i))
        plt.close(fig=fig)