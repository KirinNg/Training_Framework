from utils import config, data_stream
from utils import utils
import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

Cifar_train_config = config.Cifar_config()

Cifar_common_trainer = utils.Common_Framework(utils.cifar_net, Cifar_train_config)
Cifar_common_trainer.bulid_net()

Cifar_datastream = data_stream.Cifar_datastream(Cifar_common_trainer.sess, Cifar_train_config.BATCH_SIZE)
Cifar_datastream_val = data_stream.Cifar_datastream(Cifar_common_trainer.sess, Cifar_train_config.BATCH_SIZE, "val")


def StepLL_for_this_statu(x, logits, origin_label, eps=0.05):
    total_class_num = tf.shape(logits)[1]
    ori_class = tf.argmax(origin_label, 1, output_type=tf.int32)
    one_hot_class = tf.one_hot((ori_class + 1)//total_class_num, total_class_num)
    cross_entropy = tf.losses.softmax_cross_entropy(one_hot_class,
                                                    logits,
                                                    label_smoothing=0.1,
                                                    weights=1.0)
    x_adv = x - eps * tf.sign(tf.gradients(cross_entropy, x)[0])
    x_adv = tf.clip_by_value(x_adv, -1.0, 1.0)
    return tf.stop_gradient(x_adv), one_hot_class


attack_op = StepLL_for_this_statu(Cifar_common_trainer.image_input_placehoder,
                                  Cifar_common_trainer.logits,
                                  Cifar_common_trainer.label_input_placehoder)

for e in range(Cifar_train_config.epoch):
    for s in range(Cifar_train_config.step_per_epoch):
        image_batch, label_batch = Cifar_datastream.get_one_batch(Cifar_common_trainer.sess)

        # get attacked img
        new_batch = Cifar_common_trainer.sess.run(attack_op, feed_dict=Cifar_common_trainer.get_feed(image_batch, label_batch, "att"))

        _acc, cls_loss, l2 = Cifar_common_trainer.train(new_batch[0], new_batch[1])
        if (s + 1) % 20 == 0:
            val_image_batch, val_label_batch = Cifar_datastream_val.get_one_batch(Cifar_common_trainer.sess)
            val_acc = Cifar_common_trainer.get_acc(val_image_batch, val_label_batch)
            print("CLS_LOSS:{:.3f}, L2:{:.3f}, ACC:{:.4f}, EPOCH:{}, STEP:{}, val_ACC:{:.3f}".format(cls_loss, l2, _acc, e, s, val_acc))

Cifar_common_trainer.save("models/COM_Cifar.ckpt")
