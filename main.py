import pandas as pd
import tensorflow as tf
import time

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
    'Ticket': {'default': '', 'type': 'str'}
}
label_name = 'Survived'

# 特征预处理
feature_column_spec = {
    # 连续值, 只进入deep
    'num': [
        {'feature': 'Age'},
        {'feature': 'SibSp'},
        {'feature': 'Parch'},
        {'feature': 'Fare'}
    ],

    # 类别，onehot进入wide完成记忆，embedding进入deep完成扩展
    'cate': [
        {'feature': 'Sex', 'vocab': ['male', 'female'], 'embedding': 10},
        {'feature': 'Pclass', 'vocab': [1, 2, 3], 'embedding': 10},
        {'feature': 'Embarked', 'vocab': ['S', 'C', 'Q'], 'embedding': 10},
    ],
    'hash': [
        {'feature': 'Ticket', 'bucket': 10, 'embedding': 10}
    ],
    'bucket': [
        {'feature': 'Age', 'boundaries': [10, 20, 30, 40, 50, 60, 70, 80, 90], 'embedding': 10}
    ],

    # 人工交叉：进入wide
    'cross': [
        {'feature': ['Age#bucket', 'Sex#cate'], 'bucket': 10}
    ]
}

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

def build_feature_columns():
    num_feature_arr = []
    onehot_feature_arr = []
    embedding_feature_arr = []

    base_cate_map = {}
    for num_feature in feature_column_spec['num']:
        num_feature_arr.append(tf.feature_column.numeric_column(num_feature['feature']))
    for cate_feature in feature_column_spec['cate']:
        base_cate_map[cate_feature['feature'] + '#cate'] = (tf.feature_column.categorical_column_with_vocabulary_list(cate_feature['feature'], cate_feature['vocab']), cate_feature)
    for hash_feature in feature_column_spec['hash']:
        base_cate_map[hash_feature['feature'] + '#hash'] = (tf.feature_column.categorical_column_with_hash_bucket(hash_feature['feature'], hash_feature['bucket']), hash_feature)
    for bucket_feature in feature_column_spec['bucket']:
        num_feature = tf.feature_column.numeric_column(bucket_feature['feature'])
        base_cate_map[bucket_feature['feature'] + '#bucket'] = (tf.feature_column.bucketized_column(num_feature, boundaries=bucket_feature['boundaries']), bucket_feature)

    cross_cate_map = {}
    for cross_feature in feature_column_spec['cross']:
        cols = []
        for col_name in cross_feature['feature']:
            column, spec = base_cate_map[col_name]
            cols.append(column)
        cross_cate_map['&'.join(cross_feature['feature']) + '#cross'] = tf.feature_column.crossed_column(cols, hash_bucket_size=cross_feature['bucket'])

    for cate_name in base_cate_map:
        column, spec = base_cate_map[cate_name]
        onehot_feature_arr.append(tf.feature_column.indicator_column(column))
        embedding_feature_arr.append(tf.feature_column.embedding_column(column, spec['embedding']))
    for cross_cate_name in cross_cate_map:
        cross_feature_col = cross_cate_map[cross_cate_name]
        onehot_feature_arr.append(tf.feature_column.indicator_column(cross_feature_col))

    return onehot_feature_arr, num_feature_arr + embedding_feature_arr

def build_model():
    linear_features, dnn_features = build_feature_columns()

    input_layer = {}
    for feature_name in feature_spec:
        if feature_name == label_name:
            continue
        spec = feature_spec[feature_name]
        if spec['type'] == 'int':
            input_feature = tf.keras.Input((), name=feature_name, dtype=tf.int64)
        elif spec['type'] == 'float':
            input_feature = tf.keras.Input((), name=feature_name, dtype=tf.float32)
        elif spec['type'] == 'str':
            input_feature = tf.keras.Input((), name=feature_name, dtype=tf.string)
        input_layer[feature_name] = input_feature

    linear_feature_layer = tf.keras.layers.DenseFeatures(linear_features)
    linear_dense_layer1 = tf.keras.layers.Dense(units=1)
    output = linear_feature_layer(input_layer)
    output = linear_dense_layer1(output)
    linear_model = tf.keras.Model(inputs=list(input_layer.values()), outputs=[output])
    linear_optimizer = tf.keras.optimizers.Ftrl(l1_regularization_strength=0.5)

    dnn_feature_layer = tf.keras.layers.DenseFeatures(dnn_features)
    dnn_norm_layer = tf.keras.layers.BatchNormalization()   # important for deep
    dnn_dense_layer1 = tf.keras.layers.Dense(units=128, activation='relu')
    dnn_dense_layer2 =  tf.keras.layers.Dense(units=1)
    output = dnn_feature_layer(input_layer)
    output = dnn_norm_layer(output) # this will break the tensorboard graph because of unfixed bug
    output = dnn_dense_layer1(output)
    output = dnn_dense_layer2(output)
    dnn_model =  tf.keras.Model(inputs=list(input_layer.values()), outputs=[output])
    dnn_optimizer = tf.keras.optimizers.Adam()

    wide_deep_model = tf.keras.experimental.WideDeepModel(linear_model, dnn_model, activation='sigmoid')
    wide_deep_model.compile(optimizer=[linear_optimizer,dnn_optimizer],
                            loss=tf.keras.losses.BinaryCrossentropy(),
                            metrics=tf.keras.metrics.BinaryAccuracy())
    return wide_deep_model

def train_model(model, dataset):
    version = int(time.time())
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./tensorboard/{}'.format(version), histogram_freq=1)    # tensorboard --logdir=./tensorboard
    model.fit(dataset.batch(100).shuffle(100), epochs=2000, callbacks=[tensorboard_callback])
    model.save('model/{}'.format(version), save_format='tf', include_optimizer=False) # two optimizer in wide&deep can not be serialized, excluding optimizer is ok for prediction

# csv转tfrecords文件
csv_to_tfrecords('train.csv', 'train.tfrecords')
# 加载tfrecords文件为dataset
dataset = tfrecords_to_dataset('train.tfrecords')
# 创建wide&deep model
wide_deep_model = build_model()
# 训练模型
train_model(wide_deep_model, dataset)