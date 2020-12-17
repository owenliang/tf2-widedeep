import pandas as pd
import tensorflow as tf

# 模型输入/输出
feature_spec = {
    'Survived': {'default': -1, 'type': 'int'},
    'Pclass': {'default': -1, 'type': 'int'},
    'Sex': {'default': '', 'type': 'str'},
    'Age': {'default': -1, 'type': 'int'},
    'SibSp': {'default': -1, 'type': 'int'},
    'Parch': {'default': -1, 'type': 'int'},
    'Fare': {'default': -1, 'type': 'float'},
    'Embarked': {'default': '', 'type': 'str'},
}
label_name = 'Survived'

def csv_to_tfrecords(csv, tfrecords):
    df = pd.read_csv(csv)
    with tf.io.TFRecordWriter(tfrecords, 'GZIP') as tfrecords_writer:
        for _, row in df.iterrows():
            feature_dict = {}
            for feature_name in feature_spec:
                feature_value = row[feature_name]
                spec = feature_spec[feature_name]
                if feature_value is None or pd.isna(feature_value):
                    feature_value = spec['default']
                if spec['type'] == 'int':
                    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(feature_value)]))
                elif spec['type'] == 'float':
                    feature = tf.train.Feature(float_list=tf.train.FloatList(value=[float(feature_value)]))
                elif spec['type'] == 'str':
                    feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(feature_value).encode('utf-8')]))
                feature_dict[feature_name] = feature
            example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
            tfrecords_writer.write(example.SerializeToString())

def tfrecords_to_dataset(tfrecords_pattern):
    feature_dict = {}
    for feature_name in feature_spec:
        spec = feature_spec[feature_name]
        if spec['type'] == 'int':
            feature = tf.io.FixedLenFeature((), tf.int64)
        elif spec['type'] == 'float':
            feature = tf.io.FixedLenFeature((), tf.float32)
        elif spec['type'] == 'str':
            feature = tf.io.FixedLenFeature((), tf.string)
        feature_dict[feature_name] = feature

    def parse_func(s):
        features = tf.io.parse_single_example(s, feature_dict)
        label = features.pop(label_name)
        return features, label

    dataset = tf.data.Dataset.list_files(tfrecords_pattern).interleave(
        lambda filename: tf.data.TFRecordDataset(filename, compression_type='GZIP').map(parse_func)
    )
    return dataset

# csv转tfrecords文件
csv_to_tfrecords('train.csv', 'train.tfrecords')
# 加载tfrecords文件为dataset
dataset = tfrecords_to_dataset('train.tfrecords')
