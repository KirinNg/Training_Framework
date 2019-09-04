from utils import config, data_stream
from utils import utils
import os

os.environ["CUDA_VISIBLE_DEVISE"] = "1"

MNIST_train_config = config.MNIST_config()

MNIST_common_trainer = utils.Common_Framework(utils.lenet_net, MNIST_train_config)
MNIST_common_trainer.bulid_net()

MNIST_datastream = data_stream.MNIST_datastream()


for e in range(MNIST_train_config.epoch):
    for s in range(MNIST_train_config.step_per_epoch):
        image_batch, label_batch = MNIST_datastream.get_one_batch(MNIST_train_config.BATCH_SIZE)
        _acc, cls_loss, l2 = MNIST_common_trainer.train(image_batch, label_batch)
        if (s + 1) % 100 == 0:
            val_image_batch, val_label_batch = MNIST_datastream.get_some_test(MNIST_train_config.BATCH_SIZE)
            val_acc = MNIST_common_trainer.get_acc(val_image_batch, val_label_batch)
            print("CLS_LOSS:{:.3f}, L2:{:.3f}, ACC:{:.4f}, EPOCH:{}, STEP:{}, val_ACC:{:.3f}".format(cls_loss, l2, _acc, e, s, val_acc))


MNIST_common_trainer.save("models_8.6/COM_MNIST.ckpt")
