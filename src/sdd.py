import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
from network_module import preprocess_module, max_pool_module, cnn_module


class SSD_Model:

    def __init__(self, img_batch):
        self.image_batch = img_batch
        self.parameters = []

    def cnn(self, name, input, kernel_shape, strides=1, padding='SAME'):
        conv_relu, params = cnn_module(name, input, kernel_shape, strides=strides, padding=padding)
        self.parameters += params
        return conv_relu

    def create_ssd_graph(self):

        # VGG-16  uses 224x224 Image
        # SSD-300 uses 300x300 Image

        img_batch = preprocess_module(self.image_batch)

        conv1_1 = self.cnn("conv1_1", img_batch, [3, 3, 3, 64])
        conv1_2 = self.cnn("conv1_2", conv1_1, [3, 3, 64, 64])
        pool1_3 = max_pool_module("pool1_3", conv1_2)

        conv2_1 = self.cnn("conv2_1", pool1_3, [3, 3, 64, 128])
        conv2_2 = self.cnn("conv2_2", conv2_1, [3, 3, 128, 128])
        pool2_3 = max_pool_module("pool2_3", conv2_2)

        conv3_1 = self.cnn("conv3_1", pool2_3, [3, 3, 128, 256])
        conv3_2 = self.cnn("conv3_2", conv3_1, [3, 3, 256, 256])
        conv3_3 = self.cnn("conv3_3", conv3_2, [3, 3, 256, 256])
        pool3_4 = max_pool_module("pool3_4", conv3_3)

        conv4_1 = self.cnn("conv4_1", pool3_4, [3, 3, 256, 512])
        conv4_2 = self.cnn("conv4_2", conv4_1, [3, 3, 512, 512])
        conv4_3 = self.cnn("conv4_3", conv4_2, [3, 3, 512, 512])
        pool4_4 = max_pool_module("pool4_4", conv4_3)

        conv5_1 = self.cnn("conv5_1", pool4_4, [3, 3, 512, 512])
        conv5_2 = self.cnn("conv5_2", conv5_1, [3, 3, 512, 512])
        conv5_3 = self.cnn("conv5_3", conv5_2, [3, 3, 512, 512])

        # VGG-16  used Stride-2 and Kernel-Size-2 Pool layer
        # SSD-300 uses Stride-1 and Kernel-Size-1 Pool layer
        pool5_4 = max_pool_module("pool5_4", conv5_3, ksize=3, strides=1)

        conv6_1 = self.cnn("conv6_1", pool5_4, [3, 3, 512, 1024])

        conv7_1 = self.cnn("conv7_1", conv6_1, [3, 3, 1024, 1024])  # (19, 19, 1024)

        conv8_1 = self.cnn("conv8_1", conv7_1, [1, 1, 1024, 256])
        conv8_2 = self.cnn("conv8_2", conv8_1, [3, 3, 256, 512], strides=2)  # (10, 10, 512)

        conv9_1 = self.cnn("conv9_1", conv8_2, [1, 1, 512, 128])
        conv9_2 = self.cnn("conv9_2", conv9_1, [3, 3, 128, 256], strides=2)  # (5, 5, 256)

        conv10_1 = self.cnn("conv10_1", conv9_2, [1, 1, 256, 128])
        conv10_2 = self.cnn("conv10_2", conv10_1, [3, 3, 128, 256], padding='VALID')  # (3, 3, 256)

        conv11_1 = self.cnn("conv11_1", conv10_2, [1, 1, 256, 128])
        conv11_2 = self.cnn("conv11_2", conv11_1, [3, 3, 128, 256], padding='VALID')  # (1, 1, 256)

        return conv11_2

    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            if i <= 25:
                # print(i, k, np.shape(weights[k]))
                sess.run(self.parameters[i].assign(weights[k]))

        # Initialize extra layers added to default VGG-16 arch.
        init = tf.variables_initializer(self.parameters[26:])
        sess.run(init)


if __name__ == '__main__':
    weights_folder = "../models/vgg-16/"
    images_folder = "../data/"

    # print("Tensor-flow version " + tf.__version__)

    x = 300
    imgs = tf.placeholder(tf.float32, [None, x, x, 3])

    ssd = SSD_Model(imgs)
    sess = tf.Session()

    ssd_graph = ssd.create_ssd_graph()
    ssd.load_weights(weights_folder + "vgg16_weights.npz", sess)

    test_img = imread(images_folder + 'laska.png', mode='RGB')
    test_img = imresize(test_img, (x, x))

    out = sess.run(ssd_graph, feed_dict={
        ssd.image_batch: [test_img]
    })[0]

    print(out.shape)
    # print(out.name)
