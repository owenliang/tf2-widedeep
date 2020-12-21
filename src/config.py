from collections import OrderedDict

model_spec = {
    'inputs': {
        'Pclass': {'default': -1, 'type': 'int'},
        'Sex': {'default': '', 'type': 'str'},
        'Age': {'default': -1, 'type': 'int'},
        'SibSp': {'default': -1, 'type': 'int'},
        'Parch': {'default': -1, 'type': 'int'},
        'Fare': {'default': -1, 'type': 'float'},
        'Embarked': {'default': '', 'type': 'str'},
        'Ticket': {'default': '', 'type': 'str'}
    },
    'outputs': {
        'Survived': {'default': -1, 'type': 'int'},
    },
    'feature_columns': {
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
}