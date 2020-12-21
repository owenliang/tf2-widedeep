import pandas as pd
import tensorflow as tf
from dataset import dataframe_to_tfrecords, dataset_from_tfrecords
from model import build_model
import time

def train_model(model, dataset):
    version = int(time.time())
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./tensorboard/{}'.format(version), histogram_freq=1)    # tensorboard --logdir=./tensorboard
    model.fit(dataset.batch(100).shuffle(100), epochs=2000, callbacks=[tensorboard_callback])
    model.save('model/{}'.format(version), save_format='tf', include_optimizer=False) # two optimizer in wide&deep can not be serialized, excluding optimizer is ok for prediction

# csv转tfrecords文件
dataframe_to_tfrecords(pd.read_csv('../data/train.csv'), 'train.tfrecords', include_outputs=True)
# 加载tfrecords文件为dataset
dataset = dataset_from_tfrecords('train.tfrecords', include_outputs=True)
# 创建wide&deep model
wide_deep_model = build_model()
# 训练模型
train_model(wide_deep_model, dataset)