"""Data loading, preprocessing and model training for the Ames Housing app.
Same logic as the notebook (Sections 1, 4, 5, 6, 7): the tuned models are re-fitted here
with the best hyperparameters found by GridSearchCV, so no pickle file is needed."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, OneHotEncoder, OrdinalEncoder,
                                   PowerTransformer, RobustScaler)
from sklearn.svm import SVR

RANDOM_STATE = 42

NONE_COLS = ['Alley', 'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
             'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
             'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']

_QUAL = ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex']
ORDINAL_MAPPING = {
    'ExterQual': _QUAL[1:], 'ExterCond': _QUAL[1:], 'HeatingQC': _QUAL[1:], 'KitchenQual': _QUAL[1:],
    'BsmtQual': _QUAL, 'BsmtCond': _QUAL, 'FireplaceQu': _QUAL,
    'GarageQual': _QUAL, 'GarageCond': _QUAL, 'PoolQC': _QUAL,
    'BsmtExposure': ['None', 'No', 'Mn', 'Av', 'Gd'],
    'BsmtFinType1': ['None', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
    'BsmtFinType2': ['None', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
    'GarageFinish': ['None', 'Unf', 'RFn', 'Fin'],
    'Functional': ['Sal', 'Sev', 'Maj2', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ'],
    'LotShape': ['IR3', 'IR2', 'IR1', 'Reg'], 'LandSlope': ['Sev', 'Mod', 'Gtl'],
    'PavedDrive': ['N', 'P', 'Y']}
ORDINAL_COLS = list(ORDINAL_MAPPING)
NEW_FEATURES = ['TotalSF', 'TotalBath', 'HouseAge', 'RemodAge', 'Qual_x_Area',
                'HasGarage', 'HasBasement', 'HasPool']
BINARY_FEATURES = ['HasGarage', 'HasBasement', 'HasPool']

# Best hyperparameters from the notebook (Section 7, GridSearchCV, 5-fold CV on the train set)
TUNED = {
    'SVR':        ('log1p',        lambda: SVR(kernel='rbf', C=100, gamma=0.0003, epsilon=0.05)),
    'ElasticNet': ('yeo_johnson',  lambda: ElasticNet(alpha=0.001, l1_ratio=0.5)),
    'Lasso':      ('yeo_johnson',  lambda: Lasso(alpha=0.0003, max_iter=10000)),
    'Ridge':      ('yeo_johnson',  lambda: Ridge(alpha=10)),
    'KNN':        ('yeo_johnson',  lambda: KNeighborsRegressor(n_neighbors=5, p=1, weights='distance')),
}
FINAL_MODEL = 'SVR'


def add_features(d):
    d = d.copy()
    d['TotalSF'] = d['TotalBsmtSF'] + d['1stFlrSF'] + d['2ndFlrSF']
    d['TotalBath'] = d['FullBath'] + 0.5 * d['HalfBath'] + d['BsmtFullBath'] + 0.5 * d['BsmtHalfBath']
    d['HouseAge'] = d['YrSold'] - d['YearBuilt']
    d['RemodAge'] = d['YrSold'] - d['YearRemodAdd']
    d['HasGarage'] = (d['GarageArea'] > 0).astype(int)
    d['HasBasement'] = (d['TotalBsmtSF'] > 0).astype(int)
    d['HasPool'] = (d['PoolArea'] > 0).astype(int)
    d['Qual_x_Area'] = d['OverallQual'] * d['GrLivArea']
    return d


def load_raw(path='train.csv'):
    df = pd.read_csv(path)
    df[NONE_COLS] = df[NONE_COLS].fillna('None')
    df['MSSubClass'] = df['MSSubClass'].astype(str)
    return df


class Bundle:
    """Everything the app needs: data, split, fitted models, metrics."""
    pass


def build(path='train.csv'):
    df = load_raw(path)
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    num_cols.remove('Id'); num_cols.remove('SalePrice')
    cat_cols = df.select_dtypes(exclude='number').columns.tolist()
    nominal_cols = [c for c in cat_cols if c not in ORDINAL_COLS]

    df_clean = add_features(df[~df['Id'].isin([524, 1299])])
    num_cols = [c for c in num_cols if c not in NEW_FEATURES] + NEW_FEATURES
    feature_cols = num_cols + nominal_cols + ORDINAL_COLS
    X = df_clean[feature_cols]
    y = np.log1p(df_clean['SalePrice'])

    bins = pd.qcut(y, q=5, labels=False)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=bins)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=bins.loc[y_temp.index])

    skew = X_train[num_cols].skew()
    skewed_cols = [c for c in skew[skew.abs() > 0.75].sort_values(ascending=False).index
                   if c not in BINARY_FEATURES]
    skew_idx = [num_cols.index(c) for c in skewed_cols]

    def log_skewed(A):
        A = np.array(A, dtype=float, copy=True)
        A[:, skew_idx] = np.log1p(A[:, skew_idx])
        return A

    def prep(kind):
        if kind == 'log1p':
            num = Pipeline([('imputer', SimpleImputer(strategy='median')),
                            ('log', FunctionTransformer(log_skewed, feature_names_out='one-to-one')),
                            ('scaler', RobustScaler())])
        else:
            num = Pipeline([('imputer', SimpleImputer(strategy='median')),
                            ('yeo_johnson', ColumnTransformer(
                                [('yj', PowerTransformer(method='yeo-johnson', standardize=False), skew_idx)],
                                remainder='passthrough')),
                            ('scaler', RobustScaler())])
        nominal = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
        ordinal = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                            ('ordinal', OrdinalEncoder(categories=[ORDINAL_MAPPING[c] for c in ORDINAL_COLS],
                                                       handle_unknown='use_encoded_value', unknown_value=-1))])
        return ColumnTransformer([('num', num, num_cols), ('nominal', nominal, nominal_cols),
                                  ('ordinal', ordinal, ORDINAL_COLS)])

    models = {}
    for name, (kind, make) in TUNED.items():
        models[name] = Pipeline([('prep', prep(kind)), ('model', make())]).fit(X_train, y_train)

    def metrics(model, Xs, ys):
        yt, yp = np.expm1(ys), np.expm1(model.predict(Xs))
        return {'R²': r2_score(yt, yp), 'RMSE ($)': float(np.sqrt(mean_squared_error(yt, yp))),
                'MAE ($)': mean_absolute_error(yt, yp)}

    rows = []
    for name, m in models.items():
        for split, (Xs, ys) in {'Validation': (X_val, y_val), 'Test': (X_test, y_test)}.items():
            rows.append({'Model': name, 'Split': split, **metrics(m, Xs, ys)})
    b = Bundle()
    b.df, b.df_clean = df, df_clean
    b.X, b.y = X, y
    b.X_train, b.X_val, b.X_test, b.y_train, b.y_val, b.y_test = X_train, X_val, X_test, y_train, y_val, y_test
    b.num_cols, b.nominal_cols, b.ordinal_cols = num_cols, nominal_cols, ORDINAL_COLS
    b.models, b.results = models, pd.DataFrame(rows)
    b.final = models[FINAL_MODEL]
    # defaults used to fill every feature the user does not type in
    b.defaults = {c: X_train[c].median() for c in num_cols}
    b.defaults.update({c: X_train[c].mode().iloc[0] for c in nominal_cols + ORDINAL_COLS})
    return b


def permutation(b, n_repeats=3):
    r = permutation_importance(b.final, b.X_val, b.y_val, n_repeats=n_repeats,
                               random_state=RANDOM_STATE, scoring='neg_root_mean_squared_error')
    return (pd.Series(r.importances_mean, index=b.X_val.columns).sort_values(ascending=False))


def elasticnet_coefs(b):
    m = b.models['ElasticNet']
    names = [n.split('__')[-1] for n in m.named_steps['prep'].get_feature_names_out()]
    return pd.Series(m.named_steps['model'].coef_, index=names)


def predict_price(b, user_inputs: dict, model_name=FINAL_MODEL):
    """user_inputs: the few fields typed by the user. Everything else = training defaults."""
    row = dict(b.defaults)
    row.update(user_inputs)
    # derived raw columns kept consistent with what the user typed
    row['GrLivArea'] = row['1stFlrSF'] + row['2ndFlrSF'] + row.get('LowQualFinSF', 0)
    cars = int(row['GarageCars'])
    gdf = b.X_train[b.X_train['GarageCars'] == cars]['GarageArea']
    row['GarageArea'] = float(gdf.median()) if len(gdf) else 0.0
    if cars == 0:
        for c in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
            row[c] = 'None'
        row['GarageYrBlt'] = np.nan
    else:
        row['GarageYrBlt'] = row['YearBuilt']
    if row['TotalBsmtSF'] == 0:
        for c in ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']:
            row[c] = 'None'
        for c in ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'BsmtFullBath', 'BsmtHalfBath']:
            row[c] = 0
    else:
        row['BsmtUnfSF'] = row['TotalBsmtSF']; row['BsmtFinSF1'] = 0; row['BsmtFinSF2'] = 0
    one = add_features(pd.DataFrame([row]))[b.X_train.columns]
    return float(np.expm1(b.models[model_name].predict(one))[0])
