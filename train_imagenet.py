from utils import config, data_stream
from utils import utils
import numpy as np
import os
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

ImageNet_train_config = config.ImageNet_config()

ImageNet_common_trainer = utils.Common_Framework(utils.Alex_net, ImageNet_train_config)
ImageNet_common_trainer.bulid_net()

ImageNet_datastream = data_stream.ImageNet_datastream(ImageNet_common_trainer.sess, ImageNet_train_config.BATCH_SIZE)

# ImageNet_common_trainer.load("/home/kirin/Documents/save_model_tmp/ImageNet/imagenet.ckpt")
current_step = ImageNet_common_trainer.sess.run(ImageNet_common_trainer.global_steps)

for e in range(current_step//ImageNet_train_config.step_per_epoch, ImageNet_train_config.epoch):
    for s in range(ImageNet_train_config.step_per_epoch):
        image_batch, label_batch = ImageNet_datastream.get_one_batch_train(ImageNet_common_trainer.sess)
        _acc, _acc5, cls_loss, l2 = ImageNet_common_trainer.train(image_batch, label_batch)

        if (s + 1) % 20 == 0:
            val_image_batch, val_label_batch = ImageNet_datastream.get_one_batch_val(ImageNet_common_trainer.sess)
            val_acc, val_acc5 = ImageNet_common_trainer.get_acc(val_image_batch, val_label_batch)
            print("CLS_LOSS:{:.3f}, L2:{:.3f}, ACC:{:.4f}, ACC5:{:.4f}, EPOCH:{}, STEP:{}, VAL_ACC:{:.4f}, VAL_ACC5:{:.4f}".format(cls_loss, l2, _acc, _acc5, e, s, val_acc, val_acc5))
    ImageNet_common_trainer.save("/home/kirin/Documents/save_model_tmp/ImageNet/imagenet.ckpt")
