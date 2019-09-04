from utils import config, data_stream
from utils import utils
import os
import utils.attack_method as attack_method
import tensorflow as tf
import numpy as np

os.environ["CUDA_VISIBLE_DEVISE"] = "0,1"

MNIST_train_config = config.MNIST_config()
MNIST_train_config.BATCH_SIZE = 500

MNIST_ADD_trainer = utils.AAD_Framework(utils.lenet_net, attack_method.StepLL, MNIST_train_config)
MNIST_ADD_trainer.bulid_net()

MNIST_datastream = data_stream.MNIST_datastream()

# path = "models_8.6/COM_MNIST.ckpt"
# path = "models_8.6/Mad_MNIST.ckpt"
# path = "models_8.6/ADD_MNIST.ckpt"

path = "models_8.6/ADD4_MNIST.ckpt"

print(path)
MNIST_ADD_trainer.load(path)

# models_mnist/COM_MNIST.ckpt
# origin:0.9890000015497208
# StepLL:0.9775000005960465
# Iter_LL:0.7495999932289124
# R_StepLL:0.986700001358986
# DeepFool:0.8598999977111816

# models_mnist/Mad_MNIST.ckpt
# origin:0.9902999997138977
# StepLL:0.9874000072479248
# Iter_LL:0.9392999976873397
# R_StepLL:0.9909000009298324
# DeepFool:0.8558000028133392

# models_mnist/ADD_MNIST.ckpt
# origin:0.991799995303154
# StepLL:0.9898000001907349
# Iter_LL:0.9754000067710876
# R_StepLL:0.991400009393692
# DeepFool:0.8688999980688095

# # eval
total = 10000
iter = total//MNIST_train_config.BATCH_SIZE
count = 0
for i in range(iter):
    image_batch, label_batch = MNIST_datastream.get_some_test(MNIST_train_config.BATCH_SIZE)
    val_acc = MNIST_ADD_trainer.get_acc(image_batch, label_batch)
    count += val_acc
print("origin:{}".format(count/iter))

# stepLL
count = 0
StepLL_data = MNIST_ADD_trainer.attacked_img
for i in range(iter):
    image_batch, label_batch = MNIST_datastream.get_some_test(MNIST_train_config.BATCH_SIZE)
    # creat stepLL data
    image_batch = MNIST_ADD_trainer.sess.run(StepLL_data, feed_dict={MNIST_ADD_trainer.image_input_placehoder: image_batch, MNIST_ADD_trainer.keepprob_placehoder: 1.0})
    val_acc = MNIST_ADD_trainer.get_acc(image_batch, label_batch)
    count += val_acc
print("StepLL:{}".format(count/iter))

# Iter-LL
count = 0
for i in range(iter):
    image_batch, label_batch = MNIST_datastream.get_some_test(MNIST_train_config.BATCH_SIZE)
    # creat stepLL data
    for j in range(2):
        image_batch = MNIST_ADD_trainer.sess.run(StepLL_data, feed_dict={MNIST_ADD_trainer.image_input_placehoder: image_batch, MNIST_ADD_trainer.keepprob_placehoder: 1.0})
    val_acc = MNIST_ADD_trainer.get_acc(image_batch, label_batch)
    count += val_acc
print("Iter_LL:{}".format(count/iter))

# R+StepLL
count = 0
R_Step_LL = attack_method.StepLL(MNIST_ADD_trainer.image_input_placehoder, MNIST_ADD_trainer.org1_logits, 0.075)
for i in range(iter):
    image_batch, label_batch = MNIST_datastream.get_some_test(MNIST_train_config.BATCH_SIZE)
    # creat stepLL data
    image_batch += np.sign(np.random.standard_normal(image_batch.shape)) * 0.075
    image_batch = MNIST_ADD_trainer.sess.run(R_Step_LL, feed_dict={MNIST_ADD_trainer.image_input_placehoder: image_batch, MNIST_ADD_trainer.keepprob_placehoder: 1.0})
    val_acc = MNIST_ADD_trainer.get_acc(image_batch, label_batch)
    count += val_acc
print("R_StepLL:{}".format(count/iter))


# deepfool
# count = 0
# for i in range(iter):
#     image_batch, label_batch = MNIST_datastream.get_some_test(MNIST_train_config.BATCH_SIZE)
#     # creat DeepFool data
#     image_batch = attack_method.deepfool_attack(MNIST_ADD_trainer.sess, MNIST_ADD_trainer.image_input_placehoder,
#                                                 MNIST_ADD_trainer.org1_probs, MNIST_ADD_trainer.org1_logits, image_batch, feed={MNIST_ADD_trainer.keepprob_placehoder: 1.0})
#     val_acc = MNIST_ADD_trainer.get_acc(image_batch, label_batch)
#     count += val_acc
# print("DeepFool:{}".format(count/iter))
