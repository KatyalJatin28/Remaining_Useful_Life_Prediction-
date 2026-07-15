import pandas as pd

def load_data():
    columns = [
        'unit_id', 'cycle',
        'op_setting_1', 'op_setting_2', 'op_setting_3',
        's1', 's2', 's3', 's4', 's5',
        's6', 's7', 's8', 's9', 's10',
        's11', 's12', 's13', 's14', 's15',
        's16', 's17', 's18', 's19', 's20', 's21'
    ]

    train = pd.read_csv(
        'data/train_FD001.txt',
        sep=r'\s+',
        header=None,
        names=columns
    )

    test = pd.read_csv(
        'data/test_FD001.txt',
        sep=r'\s+',
        header=None,
        names=columns
    )

    rul = pd.read_csv(
        'data/RUL_FD001.txt',
        sep=r'\s+',
        header=None,
        names=['RUL']
    )

    return train, test, rul


def add_rul(train):
    max_cycles = train.groupby('unit_id')['cycle'].max().reset_index()
    max_cycles.columns = ['unit_id', 'max_cycle']
    train = train.merge(max_cycles, on='unit_id')
    train['RUL'] = train['max_cycle'] - train['cycle']
    train = train.drop(columns=['max_cycle'])

    return train