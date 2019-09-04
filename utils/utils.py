import tensorflow as tf
import tensorflow.contrib.slim as slim
from utils.model import *

# common_train
class Common_Framework:
    def __init__(self, net, train_config):
        self.net = net
        self.train_config = train_config

        self.image_input_placehoder = tf.placeholder(tf.float32, [self.train_config.BATCH_SIZE, self.train_config.image_size, self.train_config.image_size, self.train_config.channel_size])
        self.label_input_placehoder = tf.placeholder(tf.float32, [self.train_config.BATCH_SIZE, self.train_config.label_size])
        self.keepprob_placehoder = tf.placeholder(tf.float32, )

        self.sess = tf.Session(config=self.train_config.sess_config)

        self.global_steps = tf.Variable(0, trainable=False)


    def bulid_net(self):
        # bulid net
        self.logits, self.probs, self.end_point = self.net(self.image_input_placehoder, keep_prop=self.keepprob_placehoder)

        # cal loss
        self.classification_loss = slim.losses.softmax_cross_entropy(self.logits, self.label_input_placehoder)

        # l2 loss
        try:
            self.l2_loss = tf.add_n(slim.losses.get_regularization_losses())
        except:
            self.l2_loss = tf.constant(0, dtype=tf.float32)

        # total loss
        self.total_loss = self.classification_loss + self.l2_loss

        # useful tool
        correct_p = tf.equal(tf.argmax(self.probs, 1), (tf.argmax(self.label_input_placehoder, 1)))
        self.accuracy = tf.reduce_mean(tf.cast(correct_p, "float"))

        # ready for train
        if self.train_config.lr == "decay":
            self.lr = tf.train.exponential_decay(0.002, self.global_steps, self.train_config.step_per_epoch // 2, 0.98, staircase=False)
        else:
            self.lr = self.train_config.lr

        self.train_op = tf.train.RMSPropOptimizer(self.lr)
        self.var_list = tf.trainable_variables()

        gradients = tf.gradients(self.total_loss, self.var_list)
        self.gradients = list(zip(gradients, self.var_list))

        # Create optimizer and apply gradient descent to the trainable variables
        self.train_op = self.train_op.apply_gradients(grads_and_vars=self.gradients, global_step=self.global_steps)

        self.start_record()
        self.sess.run(tf.global_variables_initializer())

    def train(self, batch_image, batch_label):
        _acc, cls_loss, l2_loss, _, summary, gs = self.sess.run([self.accuracy, self.classification_loss, self.l2_loss, self.train_op,
                                                     self.summary_op, self.global_steps],
                                                    feed_dict=self.get_feed(batch_image, batch_label))
        self.train_writer.add_summary(summary, gs)
        return _acc, cls_loss, l2_loss

    def get_acc(self, batch_image, batch_label):
        _acc, summary, gs = self.sess.run([self.accuracy, self.summary_op, self.global_steps],
                                          feed_dict=self.get_feed(batch_image, batch_label, "val"))
        self.val_writer.add_summary(summary, gs)
        return _acc

    def get_feed(self, batch_image, batch_label, type="train"):
        if type == "train":
            return {self.image_input_placehoder: batch_image, self.label_input_placehoder: batch_label,
                    self.keepprob_placehoder: 0.5}
        else:
            return {self.image_input_placehoder: batch_image, self.label_input_placehoder: batch_label,
                    self.keepprob_placehoder: 1.0}

    def save(self, save_str):
        saver = tf.train.Saver()
        saver.save(self.sess, save_str)

    def load(self, load_path):
        saver = tf.train.Saver()
        saver.restore(self.sess, load_path)

    def start_record(self, path="log/"):
        # Add gradients to summary
        for gradient, var in self.gradients:
            tf.summary.histogram(var.name + '/gradient', gradient)

        # Add the variables we train to the summary
        for var in self.var_list:
            tf.summary.histogram(var.name, var)

        self.train_writer = tf.summary.FileWriter(path + "train", self.sess.graph)
        self.val_writer = tf.summary.FileWriter(path + "val", self.sess.graph)

        with tf.name_scope("loss"):
            tf.summary.scalar('total_loss', self.total_loss)
            tf.summary.scalar('classification_loss', self.classification_loss)
            tf.summary.scalar('l2_loss', self.l2_loss)

        tf.summary.scalar('learnning rate', self.lr)
        tf.summary.scalar('accuracy', self.accuracy)
        self.summary_op = tf.summary.merge_all()


# train_use_ADD_Framework
class AAD_Framework:
    def __init__(self, net, attack_method, train_config):
        self.net = net
        self.attack_method = attack_method
        self.train_config = train_config

        self.image_input_placehoder = tf.placeholder(tf.float32,
                                                     [self.train_config.BATCH_SIZE, self.train_config.image_size,
                                                      self.train_config.image_size, self.train_config.channel_size])
        self.adv_image_input_placehoder = tf.placeholder(tf.float32,
                                                     [self.train_config.BATCH_SIZE, self.train_config.image_size,
                                                      self.train_config.image_size, self.train_config.channel_size])
        self.label_input_placehoder = tf.placeholder(tf.float32,
                                                     [self.train_config.BATCH_SIZE, self.train_config.label_size])

        self.keepprob_placehoder = tf.placeholder(tf.float32, )

        self.sess = tf.Session(config=self.train_config.sess_config)

        self.global_steps = tf.Variable(0, trainable=False)

    def grad_cam(self, end_point, prob, layer_name='pool2', imagesize=32):
        pre_calss_one_hot = tf.one_hot(tf.argmax(prob, axis=1), self.train_config.label_size, 1.0, 0.0, dtype=tf.float32)
        conv_layer = end_point[layer_name]
        signal = tf.multiply(end_point['Logits'], pre_calss_one_hot)
        loss = tf.reduce_mean(signal, 1)
        grads = tf.gradients(loss, conv_layer)[0]
        norm_grads = tf.div(grads, tf.sqrt(tf.reduce_mean(tf.square(grads))) + tf.constant(1e-5))
        weights = tf.reduce_mean(norm_grads, axis=(1, 2))
        weights = tf.expand_dims(weights, 1)
        weights = tf.expand_dims(weights, 1)
        weights = tf.tile(weights, [1, self.train_config.gradcam_layer_size, self.train_config.gradcam_layer_size, 1])
        pre_cam = tf.multiply(weights, conv_layer)
        cam = tf.reduce_sum(pre_cam, 3)
        cam = tf.expand_dims(cam, 3)

        # norm
        cam = tf.reshape(cam, [-1, self.train_config.gradcam_layer_size * self.train_config.gradcam_layer_size])
        cam = tf.nn.softmax(cam)
        cam = tf.reshape(cam, [-1, self.train_config.gradcam_layer_size, self.train_config.gradcam_layer_size, 1])

        # cam = tf.nn.relu(cam)
        resize_cam = tf.image.resize_images(cam, [imagesize, imagesize])
        # each_max = tf.reduce_max(resize_cam, axis=[1, 2, 3])
        # each_max = tf.expand_dims(each_max, 1)
        # each_max = tf.expand_dims(each_max, 1)
        # each_max = tf.expand_dims(each_max, 1)
        # resize_cam = resize_cam / each_max
        return resize_cam, cam

    def avg_hardmask(self, input_img, resizeed_gradcam):
        each_mean = tf.reduce_mean(resizeed_gradcam, axis=[1, 2, 3])
        each_mean = tf.expand_dims(each_mean, 1)
        each_mean = tf.expand_dims(each_mean, 1)
        each_mean = tf.expand_dims(each_mean, 1)
        mask_clip = tf.sign(tf.nn.relu(resizeed_gradcam - each_mean))
        reverse_mask = tf.sign(tf.abs(mask_clip - 1))
        hole_mask = tf.tile(reverse_mask, [1, 1, 1, self.train_config.channel_size])
        return hole_mask * input_img

    def max_hardmask(self, input_img, resizeed_gradcam, alpha=0.5):
        each_max = tf.reduce_max(resizeed_gradcam, axis=[1, 2, 3])
        each_max = tf.expand_dims(each_max, 1)
        each_max = tf.expand_dims(each_max, 1)
        each_max = tf.expand_dims(each_max, 1)
        mask_clip = tf.sign(tf.nn.relu(resizeed_gradcam - each_max*alpha))
        reverse_mask = tf.sign(tf.abs(mask_clip - 1))
        hole_mask = tf.tile(reverse_mask, [1, 1, 1, self.train_config.channel_size])
        return hole_mask * input_img

    def bulid_net(self):
        # bulid net
        # up-line
        self.org1_logits, self.org1_probs, self.org1_end_point = self.net(self.image_input_placehoder, keep_prop=self.keepprob_placehoder)
        self.resized_gradcam, ori_gradcam = self.grad_cam(self.org1_end_point, self.org1_probs, layer_name=self.train_config.gradcam_layer, imagesize=self.train_config.image_size)
        self.masked_clean = self.avg_hardmask(self.image_input_placehoder, self.resized_gradcam)
        self.org2_logits, self.org2_probs, self.org2_end_point = self.net(self.masked_clean, keep_prop=self.keepprob_placehoder)

        # down-line
        self.attacked_img = self.attack_method(self.image_input_placehoder, self.org1_logits)
        self.atk1_logits, self.atk1_probs, self.atk1_end_point = self.net(self.adv_image_input_placehoder, keep_prop=self.keepprob_placehoder)
        self.atk_resized_gradcam, atk_gradcam = self.grad_cam(self.atk1_end_point, self.atk1_probs, layer_name=self.train_config.gradcam_layer, imagesize=self.train_config.image_size)
        self.masked_atk = self.avg_hardmask(self.attacked_img, self.atk_resized_gradcam)
        self.atk2_logits, self.atk2_probs, self.atk2_end_point = self.net(self.masked_atk, keep_prop=self.keepprob_placehoder)

        # cal loss
        clean_classification_loss = slim.losses.softmax_cross_entropy(self.org1_logits, self.label_input_placehoder)
        adv_classification_loss = slim.losses.softmax_cross_entropy(self.atk1_logits, self.label_input_placehoder)
        self.classification_loss = clean_classification_loss + adv_classification_loss

        # l2 loss
        try:
            self.l2_loss = tf.add_n(slim.losses.get_regularization_losses())
        except:
            self.l2_loss = tf.constant(0, dtype=tf.float32)

        # rectificationn loss
        self.rectificationn_loss = - (tf.reduce_mean(tf.square(self.org1_probs - self.org2_probs)) + tf.reduce_mean(tf.square(self.atk1_probs - self.atk2_probs)))

        # preservation loss
        self.preservation_loss = tf.reduce_mean(tf.abs(self.resized_gradcam - self.atk_resized_gradcam))
        # self.preservation_loss = tf.reduce_sum(tf.square(ori_gradcam - atk_gradcam))

        # total loss
        self.total_loss = self.train_config.alpha*(self.classification_loss + self.l2_loss) + self.train_config.beta*self.rectificationn_loss + self.train_config.gama*self.preservation_loss
        # self.total_loss = self.train_config.alpha * (self.classification_loss + self.l2_loss)

        # useful tool
        correct_p = tf.equal(tf.argmax(self.org1_probs, 1), (tf.argmax(self.label_input_placehoder, 1)))
        self.accuracy = tf.reduce_mean(tf.cast(correct_p, "float"))

        # ready for train
        self.train_op = tf.train.AdamOptimizer(self.train_config.lr).minimize(self.total_loss, global_step=self.global_steps)
        self.start_record()
        self.sess.run(tf.global_variables_initializer())

    def train(self, batch_image, batch_label, adv_batch_img):
        _acc, cls_loss, l2, p_loss, r_loss, _, self.train_summary, gs = self.sess.run([self.accuracy,
                                                                        self.classification_loss,
                                                                        self.l2_loss,
                                                                        self.preservation_loss,
                                                                        self.rectificationn_loss,
                                                                        self.train_op,
                                                                        self.summary_op,
                                                                        self.global_steps],
                                                                        feed_dict=self.get_feed(batch_image, batch_label, adv_batch_img))
        return _acc, cls_loss, l2, p_loss, r_loss

    def get_acc(self, batch_image, batch_label):
        _acc, self.val_summary, gs = self.sess.run([self.accuracy, self.summary_op, self.global_steps], feed_dict=self.get_feed(batch_image, batch_label, batch_image, type="val"))

        self.train_writer.add_summary(self.train_summary, gs)
        self.val_writer.add_summary(self.val_summary, gs)
        return _acc

    def get_feed(self, batch_image, batch_label, adv_batch_img, type="train"):
        if type == "train":
            return {self.image_input_placehoder: batch_image, self.label_input_placehoder: batch_label, self.keepprob_placehoder: 0.5, self.adv_image_input_placehoder: adv_batch_img}
        else:
            return {self.image_input_placehoder: batch_image, self.label_input_placehoder: batch_label, self.keepprob_placehoder: 1.0, self.adv_image_input_placehoder: adv_batch_img}

    def save(self, save_str):
        saver = tf.train.Saver()
        saver.save(self.sess, save_str)

    def load(self, load_path):
        saver = tf.train.Saver()
        saver.restore(self.sess, load_path)

    def start_record(self, path="log/"):
        self.train_writer = tf.summary.FileWriter(path+"train", self.sess.graph)
        self.val_writer = tf.summary.FileWriter(path+"val", self.sess.graph)

        with tf.name_scope("loss"):
            tf.summary.scalar('total_loss', self.total_loss)
            tf.summary.scalar('classification_loss', self.classification_loss)
            tf.summary.scalar('l2_loss', self.l2_loss)
            tf.summary.scalar('rectificationn_loss', self.rectificationn_loss)
            tf.summary.scalar('preservation_loss', self.preservation_loss)

        tf.summary.scalar('accuracy', self.accuracy)
        self.summary_op = tf.summary.merge_all()
