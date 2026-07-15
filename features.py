import pandas as pd

DROP_SENSORS = ['s1', 's5', 's6', 's10', 's16', 's18', 's19']
DROP_SETTINGS = ['op_setting_3']


def drop_useless(df):
    cols_to_drop = [c for c in DROP_SENSORS + DROP_SETTINGS if c in df.columns]
    return df.drop(columns=cols_to_drop)


def add_rolling_features(df, window=30):

    sensors = [
        col for col in df.columns
        if col.startswith('s') and col not in DROP_SENSORS
    ]

    for sensor in sensors:
        df[f'{sensor}_mean_{window}'] = (
            df.groupby('unit_id')[sensor]
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )
        df[f'{sensor}_std_{window}'] = (
            df.groupby('unit_id')[sensor]
            .transform(
                lambda x: x.rolling(window, min_periods=1).std().fillna(0)
            )
        )

    return df


def cap_rul(df, cap=125):
    df['RUL'] = df['RUL'].clip(upper=cap)
    return df