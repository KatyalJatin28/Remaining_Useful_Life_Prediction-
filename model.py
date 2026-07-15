import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold


def nasa_score(y_true, y_pred):
    d = np.array(y_pred) - np.array(y_true)
    score = 0
    for val in d:
        if val < 0:
            score += np.exp(-val / 13) - 1   # early prediction penalty
        else:
            score += np.exp(val / 10) - 1    # late prediction penalty (steeper)
    return score


def prepare_features(train, test_last):
    drop_cols = ['unit_id', 'cycle', 'RUL']
    feature_cols = [c for c in train.columns if c not in drop_cols]

    X_train = train[feature_cols]
    y_train = train['RUL']
    X_test = test_last[feature_cols]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, y_train, X_test_scaled, feature_cols


def train_and_evaluate(X_train, y_train, X_test, y_test):
    results = {}
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    lr_preds = np.clip(lr_preds, 0, 125)  # RUL cannot be negative

    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))
    lr_score = nasa_score(y_test, lr_preds)

    results['Linear Regression'] = {
        'model': lr,
        'predictions': lr_preds,
        'rmse': lr_rmse,
        'nasa_score': lr_score
    }

    print(f"Linear Regression  →  RMSE: {lr_rmse:.2f}  |  NASA Score: {lr_score:.2f}")

    print("\nRunning GridSearchCV for Random Forest")

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_leaf': [1, 2, 4]
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)

    grid_search = GridSearchCV(
        rf_base,
        param_grid,
        cv=kf,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(X_train, y_train)

    print(f"Best parameters:   {grid_search.best_params_}")
    print(f"Best CV RMSE:      {-grid_search.best_score_:.2f}")

    rf = grid_search.best_estimator_
    rf_preds = rf.predict(X_test)
    rf_preds = np.clip(rf_preds, 0, 125)

    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
    rf_score = nasa_score(y_test, rf_preds)

    results['Random Forest'] = {
        'model': rf,
        'predictions': rf_preds,
        'rmse': rf_rmse,
        'nasa_score': rf_score,
        'best_params': grid_search.best_params_,
        'cv_rmse': -grid_search.best_score_
    }

    print(f"Random Forest      →  RMSE: {rf_rmse:.2f}  |  NASA Score: {rf_score:.2f}")

    return results