from model import ZeroDCE
from dataset import data_generator
from glob import glob
import keras
import numpy as np
import tensorflow as tf
from PIL import Image
from plot_results import plot_loss
import config
import matplotlib.pyplot as plt

zero_model = ZeroDCE(shape=(None, None, 3))
zero_model.compile(learning_rate=1e-4)


def train(save_path=config.SAVE_WEIGHT_PATH, e=1):
    print("----------Loading training data----------")
    train_low_light_images = sorted(glob(config.TRAIN_PATH))
    val_low_light_images = sorted(glob(config.EVAL_PATH))

    train_dataset = data_generator(train_low_light_images)
    val_dataset = data_generator(val_low_light_images)

    print("Train Dataset:", train_dataset)
    print("Validation Dataset:", val_dataset)
    print("----------Training----------")
    history = zero_model.fit(train_dataset, validation_data=val_dataset, epochs=e)
    zero_model.save_weights(save_path)
    plot_loss(history, history.history)


def test(original_image):
    zero_model.load_weights(config.SAVE_WEIGHT_PATH)
    original_image = Image.open(original_image)
    image = keras.preprocessing.image.img_to_array(original_image)
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    output_image = zero_model(image)
    output_image = tf.cast((output_image[0, :, :, :] * 255), dtype=np.uint8)
    output_image = Image.fromarray(output_image.numpy())

    return output_image, original_image


if __name__ == "__main__":
    choice = 2
    if choice == 1:
        train()
    elif choice == 2:
        out, org = test("test_images/z3717134677036_8c4e9d31ee928b66922c5333e616cbd9.jpg")
        plt.subplot(121)
        plt.imshow(org)
        plt.title("Low-light image")
        plt.axis("off")

        plt.subplot(122)
        plt.imshow(out)
        plt.title("Enhanced image")
        plt.axis("off")
        plt.show()
