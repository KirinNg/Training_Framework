import tensorflow.examples.tutorials.mnist.input_data as input_data
import pickle
import numpy as np
import tensorflow as tf
import os
import tensorflow.contrib.slim as slim


class MNIST_datastream:
    def __init__(self):
        self.mnist = input_data.read_data_sets("dataset/MNIST/", one_hot=True)

    def get_one_batch(self, batch_size):
        batch = self.mnist.train.next_batch(batch_size)
        return np.reshape(batch[0], [batch_size, 28, 28, 1]), batch[1]

    def get_some_test(self, batch_size):
        batch = self.mnist.test.next_batch(batch_size)
        return np.reshape(batch[0], [batch_size, 28, 28, 1]), batch[1]


class Cifar_datastream:
    def __init__(self, sess, batch_size, mode="train", data_creat=True):
        self.mode = mode
        self.data_creat = data_creat
        features_placeholder = tf.placeholder(tf.float32, [None, 32, 32, 3])
        labels_placeholder = tf.placeholder(tf.int32, [None])

        dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, tf.one_hot(labels_placeholder, 10)))
        dataset = dataset.shuffle(20).batch(batch_size).repeat()

        iterator = dataset.make_initializable_iterator()
        self.data_element = iterator.get_next()

        if mode == "train":
            train_cifar_image = []
            train_cifar_label = []
            train_cifar_image_name = []

            for i in range(1, 6):
                cifar_file = "dataset/Cifar/cifar-10-batches-py/data_batch_" + str(i)
                print("Reading...{}".format("data_batch_" + str(i)))
                cifar = self.unpickle(cifar_file)

                cifar_label = cifar[b'labels']
                cifar_image = cifar[b'data']
                cifar_image_name = cifar[b'filenames']
                cifar_image = cifar_image.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")

                cifar_label = np.array(cifar_label)
                cifar_image_name = np.array(cifar_image_name)

                train_cifar_image.extend(cifar_image)
                train_cifar_label.extend(cifar_label)
                train_cifar_image_name.extend(cifar_image_name)

                sess.run(iterator.initializer,
                         feed_dict={features_placeholder: train_cifar_image, labels_placeholder: train_cifar_label})
        else:
            test_cifar_image = []
            test_cifar_label = []
            test_cifar_image_name = []

            print("Reading...{}".format("test_set"))
            cifar_file = "dataset/Cifar/cifar-10-batches-py/test_batch"
            cifar = self.unpickle(cifar_file)

            cifar_label = cifar[b'labels']
            cifar_image = cifar[b'data']
            cifar_image_name = cifar[b'filenames']
            cifar_image = cifar_image.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")

            test_cifar_image.extend(cifar_image)
            test_cifar_label.extend(cifar_label)
            test_cifar_image_name.extend(cifar_image_name)

            sess.run(iterator.initializer,
                     feed_dict={features_placeholder: test_cifar_image, labels_placeholder: test_cifar_label})

        if self.mode == "train" and self.data_creat:
            img = tf.image.random_flip_left_right(self.data_element[0])
            img = tf.image.random_brightness(img, max_delta=63)
            img = tf.image.random_contrast(img, lower=0.2, upper=1.8)
            self.data_img = (img/255 - 0.5) * 2
            self.data_label = self.data_element[1]
        else:
            self.data_img = (self.data_element[0]/255 - 0.5) * 2
            self.data_label = self.data_element[1]

    def unpickle(self, file):
        with open(file, 'rb') as f:
            cifar_dict = pickle.load(f, encoding='bytes')
        return cifar_dict

    def get_one_batch(self, sess):
        image, label = sess.run([self.data_img, self.data_label])
        return image, label


class ImageNet_datastream:
    def __init__(self, sess, batchsize=10):
        self.train_img_batch, self.train_label_batch = self.read_and_decode("/home/kirin/Python_Code/#DATASET/tfrecord", "train", batchsize)
        self.val_img_batch, self.val_label_batch = self.read_and_decode("/home/kirin/Python_Code/#DATASET/tfrecord", "val", batchsize)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)

    def read_and_decode(self, path, type="train", batchsize=10, imgsize=224):
        if type == "train":
            file_path = os.path.join(path, "train-*")
            num_samples = 1281167

            dataset = self.get_record_dataset(file_path, num_samples=num_samples, num_classes=1000)
            data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset)
            image, label = data_provider.get(['image', 'label'])

            image = self._fixed_sides_resize(image, output_height=imgsize, output_width=imgsize)

            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, max_delta=0.1)
            image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
            image = image / 255

            img_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batchsize, num_threads=32,
                                                            capacity=8192*4, min_after_dequeue=512)
            label_batch = tf.one_hot(label_batch, 1000)
            return img_batch, label_batch
        else:
            file_path = os.path.join(path, "validation-*")
            num_samples = 50000

            dataset = self.get_record_dataset(file_path, num_samples=num_samples, num_classes=1000)
            data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset)
            image, label = data_provider.get(['image', 'label'])

            image = self._fixed_sides_resize(image, output_height=imgsize, output_width=imgsize)

            image = image / 255

            # img_batch, label_batch = tf.train.batch([image, label], batch_size=batchsize, allow_smaller_final_batch=True)
            img_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batchsize, num_threads=2,
                                                            capacity=4096, min_after_dequeue=512)
            label_batch = tf.one_hot(label_batch, 1000)
            return img_batch, label_batch


    def _fixed_sides_resize(self, image, output_height, output_width):
        """Resize images by fixed sides.

        Args:
            image: A 3-D image `Tensor`.
            output_height: The height of the image after preprocessing.
            output_width: The width of the image after preprocessing.
        Returns:
            resized_image: A 3-D tensor containing the resized image.
        """
        output_height = tf.convert_to_tensor(output_height, dtype=tf.int32)
        output_width = tf.convert_to_tensor(output_width, dtype=tf.int32)

        image = tf.expand_dims(image, 0)
        resized_image = tf.image.resize_nearest_neighbor(
            image, [output_height, output_width], align_corners=False)
        resized_image = tf.squeeze(resized_image)
        resized_image.set_shape([None, None, 3])
        return resized_image

    def get_record_dataset(self, record_path, reader=None, num_samples=1281167, num_classes=1000):
        """Get a tensorflow record file.

        Args:

        """
        if not reader:
            reader = tf.TFRecordReader

        keys_to_features = {
            'image/encoded':
                tf.FixedLenFeature((), tf.string, default_value=''),
            'image/format':
                tf.FixedLenFeature((), tf.string, default_value='jpeg'),
            'image/class/label':
                tf.FixedLenFeature([1], tf.int64, default_value=tf.zeros([1],
                                                                         dtype=tf.int64))}

        items_to_handlers = {
            'image': slim.tfexample_decoder.Image(image_key='image/encoded',
                                                  format_key='image/format'),
            'label': slim.tfexample_decoder.Tensor('image/class/label', shape=[])}
        decoder = slim.tfexample_decoder.TFExampleDecoder(
            keys_to_features, items_to_handlers)

        labels_to_names = None
        items_to_descriptions = {
            'image': 'An image with shape image_shape.',
            'label': 'A single integer.'}
        return slim.dataset.Dataset(
            data_sources=record_path,
            reader=reader,
            decoder=decoder,
            num_samples=num_samples,
            num_classes=num_classes,
            items_to_descriptions=items_to_descriptions,
            labels_to_names=labels_to_names)

    def get_one_batch_train(self, sess):
        image, label = sess.run([self.train_img_batch, self.train_label_batch])
        return image, label

    def get_one_batch_val(self, sess):
        image, label = sess.run([self.val_img_batch, self.val_label_batch])
        return image, label

