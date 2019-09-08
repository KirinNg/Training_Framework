import tensorflow as tf


# config_cifar
class Cifar_config:
    def __init__(self):
        # train config
        self.lr = 0.0001
        self.sess_config = tf.ConfigProto()
        self.sess_config.gpu_options.allow_growth = True
        self.BATCH_SIZE = 100

        # dataset config
        self.image_size = 32
        self.label_size = 10
        self.channel_size = 3

        # step_train
        self.epoch = 200
        self.step_per_epoch = 50000 // self.BATCH_SIZE

        # AAD config
        self.gradcam_layer = "pool2"
        self.gradcam_layer_size = 8
        self.gradcam_layer_channel = 64
        self.alpha = 8
        self.beta = 4
        self.gama = 1


# config_mnist
class MNIST_config:
    def __init__(self):
        # train config
        self.lr = 0.001
        self.sess_config = tf.ConfigProto()
        self.sess_config.gpu_options.allow_growth = True
        self.BATCH_SIZE = 50

        # dataset config
        self.image_size = 28
        self.label_size = 10
        self.channel_size = 1

        # step_train
        self.epoch = 5
        self.step_per_epoch = 60000 // self.BATCH_SIZE

        # AAD config
        self.gradcam_layer = "pool2"
        self.gradcam_layer_size = 7
        self.gradcam_layer_channel = 64
        self.alpha = 2/5
        self.beta = 1/5
        self.gama = 2/5


# config_ImageNet
class ImageNet_config:
    def __init__(self):
        # step_train
        self.epoch = 100
        self.BATCH_SIZE = 512
        self.step_per_epoch = 1281167 // self.BATCH_SIZE

        # train config
        # self.lr = 0.04
        self.lr = "decay"
        self.sess_config = tf.ConfigProto()
        self.sess_config.gpu_options.allow_growth = True


        # dataset config
        self.image_size = 224
        self.label_size = 1000
        self.channel_size = 3

        # AAD config
        # self.gradcam_layer = "pool2"
        # self.gradcam_layer_size = 7
        # self.gradcam_layer_channel = 64
        # self.alpha = 2
        # self.beta = 1
        # self.gama = 2
