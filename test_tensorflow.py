import tensorflow as tf;
print(tf.config.list_physical_devices('GPU'));
print();
tf.test.gpu_device_name();
print();
tf.test.is_built_with_cuda();
print();
from tensorflow.python.client import device_lib;
print(device_lib.list_local_devices())