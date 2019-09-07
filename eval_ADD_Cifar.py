from utils import config, data_stream
from utils import utils
import os
import utils.attack_method as attack_method
import tensorflow as tf
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

Cifar_train_config = config.Cifar_config()
Cifar_train_config.BATCH_SIZE = 500

Cifar_ADD_trainer = utils.AAD_Framework(utils.cifar_net, attack_method.StepLL, Cifar_train_config)
Cifar_ADD_trainer.bulid_net()

# train_Cifar_datastream = data_stream.Cifar_datastream(Cifar_ADD_trainer.sess, Cifar_train_config.BATCH_SIZE, "train")
val_Cifar_datastream = data_stream.Cifar_datastream(Cifar_ADD_trainer.sess, Cifar_train_config.BATCH_SIZE, "val")

# path = "models_8.7/ADD_Cifar.ckpt"
# path = "models_8.7/COM_Cifar.ckpt"
# path = "models_8.7/Mad_Cifar.ckpt"

# path = "models/ADD_add_1_iter_200_epoches_Cifar.ckpt"
# path = "models/ADD_add_2_iter_200_epoches_Cifar.ckpt"
path = "models/ADD_add_4_iter_200_epoches_Cifar.ckpt"
# path = "models/ADD_add_5_iter_200_epoches_Cifar.ckpt"

Cifar_ADD_trainer.load(path)
print(path)

# COM
# origin:0.9927000015974045
# StepLL:0.9744999974966049
# DeepFool:0.7243999928236008

# ADD
# origin:0.9915999948978425
# StepLL:0.9879000037908554
# DeepFool:0.8409999996423722

# eval
total = 10000
iter = total//Cifar_train_config.BATCH_SIZE
count = 0
for i in range(iter):
    image_batch, label_batch = val_Cifar_datastream.get_one_batch(Cifar_ADD_trainer.sess)
    val_acc = Cifar_ADD_trainer.get_acc(image_batch, label_batch)
    count += val_acc
print("origin:{}".format(count/iter))

# stepLL
count = 0
StepLL_data = Cifar_ADD_trainer.attacked_img
for i in range(iter):
    image_batch, label_batch = val_Cifar_datastream.get_one_batch(Cifar_ADD_trainer.sess)
    # creat stepLL data
    image_batch = Cifar_ADD_trainer.sess.run(StepLL_data, feed_dict={Cifar_ADD_trainer.image_input_placehoder: image_batch, Cifar_ADD_trainer.keepprob_placehoder: 1.0})
    val_acc = Cifar_ADD_trainer.get_acc(image_batch, label_batch)
    count += val_acc
print("StepLL:{}".format(count/iter))

# Iter-LL
count = 0
for i in range(iter):
    image_batch, label_batch = val_Cifar_datastream.get_one_batch(Cifar_ADD_trainer.sess)
    # creat stepLL data
    for j in range(2):
        image_batch = Cifar_ADD_trainer.sess.run(StepLL_data, feed_dict={Cifar_ADD_trainer.image_input_placehoder: image_batch, Cifar_ADD_trainer.keepprob_placehoder: 1.0})
    val_acc = Cifar_ADD_trainer.get_acc(image_batch, label_batch)
    count += val_acc
print("Iter_LL:{}".format(count/iter))

# R+StepLL
count = 0
R_Step_LL = attack_method.StepLL(Cifar_ADD_trainer.image_input_placehoder, Cifar_ADD_trainer.org1_logits, 0.075)
for i in range(iter):
    image_batch, label_batch = val_Cifar_datastream.get_one_batch(Cifar_ADD_trainer.sess)
    # creat stepLL data
    image_batch += np.sign(np.random.standard_normal(image_batch.shape)) * 0.075
    image_batch = Cifar_ADD_trainer.sess.run(R_Step_LL, feed_dict={Cifar_ADD_trainer.image_input_placehoder: image_batch, Cifar_ADD_trainer.keepprob_placehoder: 1.0})
    val_acc = Cifar_ADD_trainer.get_acc(image_batch, label_batch)
    count += val_acc
print("R_StepLL:{}".format(count/iter))


# # deepfool
# count = 0
# for i in range(iter):
#     print(" {}".format(i/iter))
#     image_batch, label_batch = val_Cifar_datastream.get_one_batch(Cifar_ADD_trainer.sess)
#     # creat DeepFool data
#     image_batch = attack_method.deepfool_attack(Cifar_ADD_trainer.sess, Cifar_ADD_trainer.image_input_placehoder,
#                                                 Cifar_ADD_trainer.org1_probs, Cifar_ADD_trainer.org1_logits, image_batch, feed={Cifar_ADD_trainer.keepprob_placehoder: 1.0})
#     val_acc = Cifar_ADD_trainer.get_acc(image_batch, label_batch)
#     count += val_acc
# print("DeepFool:{}".format(count/iter))
