import os
import sys

import cv2
import numpy as np
import tensorflow as tf

# import network
# import guided_filter
from tqdm import tqdm

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

# import keras_cv
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, backend
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K


def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x

    return tf.numpy_function(f, [y_true, y_pred], tf.float32)


smooth = 1e-15


# smooth = 1e-2
def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth
    )


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    # img = mpimg.imread(path)
    # plt.imshow(img)
    x = x / 255.0
    x = x.astype(np.float32)
    return x


def resize_crop(image):
    h, w, c = np.shape(image)
    if min(h, w) > 720:
        if h > w:
            h, w = int(720 * h / w), 720
        else:
            h, w = 720, int(720 * w / h)
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    h, w = (h // 8) * 8, (w // 8) * 8
    image = image[:h, :w, :]
    return image


import os

# Get the model file path from environment variables
model_file_path = os.environ.get("MODEL_FILE_PATH", "./model.h5")


def cartoonize(load_folder, save_folder, model_path):
    print("c2")
    model = tf.keras.models.load_model(
        "./model.h5",
        custom_objects={"iou": iou, "dice_coef": dice_coef, "dice_loss": dice_loss},
    )
    print("c3")
    # input_photo = tf.placeholder(tf.float32, [1, None, None, 3])
    # network_out = network.unet_generator(input_photo)
    # final_out = guided_filter.guided_filter(input_photo, network_out, r=1, eps=5e-3)

    # all_vars = tf.trainable_variables()
    # gene_vars = [var for var in all_vars if "generator" in var.name]
    # saver = tf.train.Saver(var_list=gene_vars)

    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # sess = tf.Session(config=config)

    # sess.run(tf.global_variables_initializer())
    # saver.restore(sess, tf.train.latest_checkpoint(model_path))
    name_list = os.listdir(load_folder)
    for name in tqdm(name_list):
        print("c5")
        try:
            load_path = os.path.join(load_folder, name)
            save_path = os.path.join(save_folder, name)
            # image = cv2.imread(load_path)
            print("c6")
            image = read_image(load_path)
            resized_image = cv2.resize(image, (512, 256))
            print("c7")
            # image = resize_crop(image)
            # batch_image = image.astype(np.float32) / 127.5 - 1
            # batch_image = np.expand_dims(batch_image, axis=0)
            # output = sess.run(final_out, feed_dict={input_photo: batch_image})
            output = model.predict(np.expand_dims(resized_image, axis=0))[0]
            print("c8")
            output = output > 0.5
            output = output.astype(np.int32)
            print("c9")
            output = np.squeeze(output, axis=-1)
            plt.imshow(output)
            # plt.show()
            print("c10")
            # output = (np.squeeze(output) + 1) * 127.5
            # output = np.clip(output, 0, 255).astype(np.uint8)
            plt.savefig(save_path)
            print("c11")
            print()
        except:
            print("cartoonize {} failed".format(load_path))
    print()


if __name__ == "__main__":
    print("c1")
    model_path = "saved_models"
    load_folder = "public/uploads"
    save_folder = "public/pyimages"
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    print("c1.1")
    load_folder = "public/uploads"
    up_img_folder = os.listdir("public/uploads")
    py_img_folder = os.listdir("public/pyimages")
    if py_img_folder:
        for name in py_img_folder:
            os.remove("public/pyimages/" + name)
    print("c1.2")
    if up_img_folder:
        for name in up_img_folder:
            if name != sys.argv[1]:
                os.remove("public/uploads/" + name)

    cartoonize(load_folder, save_folder, model_path)
