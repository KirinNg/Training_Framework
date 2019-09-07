from utils import config, data_stream
from utils import utils
import os
import utils.attack_method as attack_method

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

Cifar_train_config = config.Cifar_config()

Cifar_AAD_trainer = utils.AAD_Framework(utils.cifar_net, attack_method.StepLL, Cifar_train_config)
Cifar_AAD_trainer.bulid_net()

Cifar_datastream = data_stream.Cifar_datastream(Cifar_AAD_trainer.sess, Cifar_train_config.BATCH_SIZE)

Cifar_datastream_val = data_stream.Cifar_datastream(Cifar_AAD_trainer.sess, Cifar_train_config.BATCH_SIZE, "val")


for e in range(Cifar_train_config.epoch):
    for s in range(Cifar_train_config.step_per_epoch):
        image_batch, label_batch = Cifar_datastream.get_one_batch(Cifar_AAD_trainer.sess)
        adv_image_batch = Cifar_AAD_trainer.sess.run(Cifar_AAD_trainer.attacked_img,
                                                     feed_dict={Cifar_AAD_trainer.image_input_placehoder: image_batch,
                                                                Cifar_AAD_trainer.keepprob_placehoder: 1.0})
        _acc, cls_loss, l2, p_loss, r_loss = Cifar_AAD_trainer.train(image_batch, label_batch, adv_image_batch)

        adv_image_batch = Cifar_AAD_trainer.sess.run(Cifar_AAD_trainer.attacked_img,
                                                     feed_dict={Cifar_AAD_trainer.image_input_placehoder: adv_image_batch,
                                                                Cifar_AAD_trainer.keepprob_placehoder: 1.0})
        _acc, cls_loss, l2, p_loss, r_loss = Cifar_AAD_trainer.train(image_batch, label_batch, adv_image_batch)

        adv_image_batch = Cifar_AAD_trainer.sess.run(Cifar_AAD_trainer.attacked_img,
                                                     feed_dict={Cifar_AAD_trainer.image_input_placehoder: adv_image_batch,
                                                                Cifar_AAD_trainer.keepprob_placehoder: 1.0})
        _acc, cls_loss, l2, p_loss, r_loss = Cifar_AAD_trainer.train(image_batch, label_batch, adv_image_batch)

        adv_image_batch = Cifar_AAD_trainer.sess.run(Cifar_AAD_trainer.attacked_img,
                                                     feed_dict={Cifar_AAD_trainer.image_input_placehoder: adv_image_batch,
                                                                Cifar_AAD_trainer.keepprob_placehoder: 1.0})
        _acc, cls_loss, l2, p_loss, r_loss = Cifar_AAD_trainer.train(image_batch, label_batch, adv_image_batch)

        adv_image_batch = Cifar_AAD_trainer.sess.run(Cifar_AAD_trainer.attacked_img,
                                                     feed_dict={Cifar_AAD_trainer.image_input_placehoder: adv_image_batch,
                                                                Cifar_AAD_trainer.keepprob_placehoder: 1.0})
        _acc, cls_loss, l2, p_loss, r_loss = Cifar_AAD_trainer.train(image_batch, label_batch, adv_image_batch)

        if (s + 1) % 100 == 0:
            val_image_batch, val_label_batch = Cifar_datastream_val.get_one_batch(Cifar_AAD_trainer.sess)
            val_acc = Cifar_AAD_trainer.get_acc(val_image_batch, val_label_batch)
            print("CLS_LOSS:{:.3f}, P_LOSS:{:.3e}, R_LOSS:{:.3e}, L2:{:.3f}, ACC:{:.4f}, EPOCH:{}, STEP:{}, val_ACC:{:.3f}".format(cls_loss, p_loss, r_loss, l2, _acc, e, s, val_acc))

Cifar_AAD_trainer.save("models/ADD_add_5_iter_200_epoches_Cifar.ckpt")

