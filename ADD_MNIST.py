from utils import config, data_stream
from utils import utils
import os
import utils.attack_method as attack_method

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

MNIST_train_config = config.MNIST_config()

MNIST_ADD_trainer = utils.AAD_Framework(utils.lenet_net, attack_method.StepLL, MNIST_train_config)
MNIST_ADD_trainer.bulid_net()

MNIST_datastream = data_stream.MNIST_datastream()


for e in range(MNIST_train_config.epoch):
    for s in range(MNIST_train_config.step_per_epoch):
        image_batch, label_batch = MNIST_datastream.get_one_batch(MNIST_train_config.BATCH_SIZE)
        adv_image_batch = MNIST_ADD_trainer.sess.run(MNIST_ADD_trainer.attacked_img,
                                                     feed_dict={MNIST_ADD_trainer.image_input_placehoder: image_batch,
                                                                MNIST_ADD_trainer.keepprob_placehoder: 1.0})
        _acc, cls_loss, l2, p_loss, r_loss = MNIST_ADD_trainer.train(image_batch, label_batch, adv_image_batch)

        adv_image_batch = MNIST_ADD_trainer.sess.run(MNIST_ADD_trainer.attacked_img,
                                                     feed_dict={MNIST_ADD_trainer.image_input_placehoder: adv_image_batch,
                                                                MNIST_ADD_trainer.keepprob_placehoder: 1.0})
        _acc, cls_loss, l2, p_loss, r_loss = MNIST_ADD_trainer.train(image_batch, label_batch, adv_image_batch)

        adv_image_batch = MNIST_ADD_trainer.sess.run(MNIST_ADD_trainer.attacked_img,
                                                     feed_dict={MNIST_ADD_trainer.image_input_placehoder: adv_image_batch,
                                                                MNIST_ADD_trainer.keepprob_placehoder: 1.0})
        _acc, cls_loss, l2, p_loss, r_loss = MNIST_ADD_trainer.train(image_batch, label_batch, adv_image_batch)

        adv_image_batch = MNIST_ADD_trainer.sess.run(MNIST_ADD_trainer.attacked_img,
                                                     feed_dict={MNIST_ADD_trainer.image_input_placehoder: adv_image_batch,
                                                                MNIST_ADD_trainer.keepprob_placehoder: 1.0})
        _acc, cls_loss, l2, p_loss, r_loss = MNIST_ADD_trainer.train(image_batch, label_batch, adv_image_batch)

        if (s + 1) % 10 == 0:
            val_image_batch, val_label_batch = MNIST_datastream.get_some_test(MNIST_train_config.BATCH_SIZE)
            val_acc = MNIST_ADD_trainer.get_acc(val_image_batch, val_label_batch)
            print("CLS_LOSS:{:.3f}, P_LOSS:{:.3e}, R_LOSS:{:.3e}, L2:{:.3f}, ACC:{:.4f}, EPOCH:{}, STEP:{}, val_ACC:{:.3f}".format(cls_loss, p_loss, r_loss, l2, _acc, e, s, val_acc))


MNIST_ADD_trainer.save("models_8.6/ADD4_MNIST.ckpt")
