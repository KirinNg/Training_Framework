from utils import config, data_stream
from utils import utils
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

Cifar_train_config = config.Cifar_config()

Cifar_common_trainer = utils.Common_Framework(utils.cifar_net, Cifar_train_config)
Cifar_common_trainer.bulid_net()

Cifar_datastream = data_stream.Cifar_datastream(Cifar_common_trainer.sess, Cifar_train_config.BATCH_SIZE)
Cifar_datastream_val = data_stream.Cifar_datastream(Cifar_common_trainer.sess, Cifar_train_config.BATCH_SIZE, "val")


for e in range(Cifar_train_config.epoch):
    for s in range(Cifar_train_config.step_per_epoch):
        image_batch, label_batch = Cifar_datastream.get_one_batch(Cifar_common_trainer.sess)
        _acc, cls_loss, l2 = Cifar_common_trainer.train(image_batch, label_batch)
        if (s + 1) % 20 == 0:
            val_image_batch, val_label_batch = Cifar_datastream_val.get_one_batch(Cifar_common_trainer.sess)
            val_acc = Cifar_common_trainer.get_acc(val_image_batch, val_label_batch)
            print("CLS_LOSS:{:.3f}, L2:{:.3f}, ACC:{:.4f}, EPOCH:{}, STEP:{}, val_ACC:{:.3f}".format(cls_loss, l2, _acc, e, s, val_acc))

Cifar_common_trainer.save("models/COM_Cifar.ckpt")
