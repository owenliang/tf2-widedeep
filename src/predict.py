import pandas as pd
import numpy as np
import tensorflow as tf
from dataset import dataframe_to_tfrecords, dataset_from_tfrecords

# 预测
def model_predict(model, dataset, df ):
    dataset = dataset.batch(100)
    pred_y = []
    for x in dataset:
        y = wide_deep_model(x)
        y = np.where(y.numpy()[:, 0]>0.5, 1, 0)
        pred_y.extend(y)
    df['Survived'] = pred_y

# csv转tfrecords文件
df = pd.read_csv('../data/test.csv')
dataframe_to_tfrecords(df, 'test.tfrecords', include_outputs=False)
# 加载tfrecords文件为dataset
dataset = dataset_from_tfrecords('test.tfrecords', include_outputs=False)
# 加载模型
wide_deep_model = tf.keras.models.load_model('model/1608529479')
# 使用模型
model_predict(wide_deep_model, dataset, df)
# 保存预测结果
df[['PassengerId','Survived']].to_csv('./result.csv', index=False, header=True)