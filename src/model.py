import tensorflow as tf
from config import model_spec

def _build_feature_columns():
    num_feature_arr = []
    onehot_feature_arr = []
    embedding_feature_arr = []

    feature_column_spec = model_spec['feature_columns']

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

def build_input_layer():
    input_layer = {}
    for input_name, input_spec in model_spec['inputs'].items():
        if input_spec['type'] == 'int':
            input_feature = tf.keras.Input((), name=input_name, dtype=tf.int64)
        elif input_spec['type'] == 'float':
            input_feature = tf.keras.Input((), name=input_name, dtype=tf.float32)
        elif input_spec['type'] == 'str':
            input_feature = tf.keras.Input((), name=input_name, dtype=tf.string)
        input_layer[input_name] = input_feature
    return input_layer

def build_model():
    linear_features, dnn_features = _build_feature_columns()

    # wide
    linear_input_layer = build_input_layer()
    linear_feature_layer = tf.keras.layers.DenseFeatures(linear_features)
    linear_dense_layer1 = tf.keras.layers.Dense(units=1)

    output = linear_feature_layer(linear_input_layer)
    output = linear_dense_layer1(output)

    linear_model = tf.keras.Model(inputs=list(linear_input_layer.values()), outputs=[output])
    linear_optimizer = tf.keras.optimizers.Ftrl(l1_regularization_strength=0.5)

    # deep
    dnn_feature_layer = tf.keras.layers.DenseFeatures(dnn_features)
    dnn_norm_layer = tf.keras.layers.BatchNormalization()   # important for deep
    dnn_dense_layer1 = tf.keras.layers.Dense(units=128, activation='relu')
    dnn_dense_layer2 =  tf.keras.layers.Dense(units=1)

    dnn_input_layer = build_input_layer()
    output = dnn_feature_layer(dnn_input_layer)
    output = dnn_norm_layer(output) # this will break the tensorboard graph because of unfixed bug
    output = dnn_dense_layer1(output)
    output = dnn_dense_layer2(output)

    dnn_model =  tf.keras.Model(inputs=list(dnn_input_layer.values()), outputs=[output])
    dnn_optimizer = tf.keras.optimizers.Adagrad()

    # wide&deep
    wide_deep_model = tf.keras.experimental.WideDeepModel(linear_model, dnn_model, activation='sigmoid')
    wide_deep_model.compile(optimizer=[linear_optimizer,dnn_optimizer],
                            loss=tf.keras.losses.BinaryCrossentropy(),
                            metrics=tf.keras.metrics.BinaryAccuracy())
    return wide_deep_model