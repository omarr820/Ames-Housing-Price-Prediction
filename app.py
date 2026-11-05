import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

import core

st.set_page_config(page_title="Ames Housing Price Prediction", page_icon="🏠", layout="wide")


@st.cache_resource(show_spinner="Training the models (a few seconds, only the first time)...")
def get_bundle():
    return core.build('train.csv')


@st.cache_data(show_spinner="Computing permutation importance...")
def get_permutation(_b):
    return core.permutation(_b)


b = get_bundle()
df = b.df_clean
res = b.results
final_test = res[(res.Model == core.FINAL_MODEL) & (res.Split == 'Test')].iloc[0]

st.sidebar.title("🏠 Ames Housing")
page = st.sidebar.radio("Page", ["💰 Price Prediction", "📊 Data Explorer",
                                 "🏆 Model Results", "🔍 Feature Importance", "ℹ️ About"])
st.sidebar.caption(f"Final model: **{core.FINAL_MODEL}** (rbf, C=100, gamma=0.0003, epsilon=0.05, log1p)")

# ---------------------------------------------------------------- Prediction
if page.startswith("💰"):
    st.title("💰 Predict a house price")
    st.write("Fill in the main features of the house. Every other feature (about 75 of the 87) is filled "
             "automatically with the typical value from the training data.")
    nb = sorted(df['Neighborhood'].unique())
    q_levels = ['Po', 'Fa', 'TA', 'Gd', 'Ex']

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Quality")
        qual = st.slider("Overall quality (1-10)", 1, 10, 6)
        cond = st.slider("Overall condition (1-9)", 1, 9, 5)
        kitchen = st.selectbox("Kitchen quality", q_levels[1:], index=2)
        exter = st.selectbox("Exterior quality", q_levels[1:], index=2)
        bsmtq = st.selectbox("Basement quality", ['None'] + q_levels, index=4)
    with c2:
        st.subheader("Size")
        f1 = st.number_input("1st floor area (sq ft)", 300, 4000, 1100, step=50)
        f2 = st.number_input("2nd floor area (sq ft)", 0, 2500, 0, step=50)
        bsmt = st.number_input("Basement area (sq ft, 0 = none)", 0, 4000, 1000, step=50)
        lot = st.number_input("Lot area (sq ft)", 1300, 60000, 9500, step=500)
        garage = st.selectbox("Garage capacity (cars)", [0, 1, 2, 3, 4], index=2)
    with c3:
        st.subheader("Location & age")
        hood = st.selectbox("Neighborhood", nb, index=nb.index('NAmes'))
        built = st.number_input("Year built", 1872, 2010, 1990)
        remod = st.number_input("Year remodeled", 1950, 2010, 1990)
        sold = st.number_input("Year sold", 2006, 2010, 2008)
        full = st.selectbox("Full bathrooms", [0, 1, 2, 3, 4], index=2)
        half = st.selectbox("Half bathrooms", [0, 1, 2], index=0)
        fire = st.selectbox("Fireplaces", [0, 1, 2, 3], index=0)

    model_name = st.selectbox("Model", list(core.TUNED), index=list(core.TUNED).index(core.FINAL_MODEL))
    user = {'OverallQual': qual, 'OverallCond': cond, 'KitchenQual': kitchen, 'ExterQual': exter,
            'BsmtQual': bsmtq, '1stFlrSF': f1, '2ndFlrSF': f2, 'TotalBsmtSF': bsmt, 'LotArea': lot,
            'GarageCars': garage, 'Neighborhood': hood, 'YearBuilt': built,
            'YearRemodAdd': max(remod, built), 'YrSold': sold, 'FullBath': full, 'HalfBath': half,
            'Fireplaces': fire}
    if st.button("Predict price", type="primary"):
        p = core.predict_price(b, user, model_name)
        rmse = float(res[(res.Model == model_name) & (res.Split == 'Test')]['RMSE ($)'].iloc[0])
        st.metric("Estimated sale price", f"${p:,.0f}")
        st.caption(f"Typical error of this model on unseen houses: about ±${rmse:,.0f} (test RMSE). "
                   f"Median price in the data: ${df['SalePrice'].median():,.0f}.")
        hood_df = df[df['Neighborhood'] == hood]['SalePrice']
        fig, ax = plt.subplots(figsize=(8, 3))
        sns.histplot(df['SalePrice'], bins=40, color='lightgray', ax=ax)
        ax.axvline(p, color='crimson', lw=2, label=f'Prediction ${p:,.0f}')
        ax.axvline(hood_df.median(), color='steelblue', ls='--', label=f'{hood} median ${hood_df.median():,.0f}')
        ax.set_xlabel('SalePrice'); ax.legend()
        st.pyplot(fig)
        if len(set(core.TUNED)) > 1:
            comp = {m: core.predict_price(b, user, m) for m in core.TUNED}
            st.write("Same house, every model:")
            st.dataframe(pd.DataFrame({'Predicted price ($)': comp}).round(0).astype(int).style.format("{:,}"))

# ---------------------------------------------------------------- Explorer
elif page.startswith("📊"):
    st.title("📊 Data Explorer")
    t1, t2, t3, t4 = st.tabs(["Price", "Neighborhoods", "Features vs price", "Correlation"])
    with t1:
        c1, c2, c3 = st.columns(3)
        c1.metric("Houses (after removing 2 outliers)", len(df))
        c2.metric("Mean price", f"${df['SalePrice'].mean():,.0f}")
        c3.metric("Median price", f"${df['SalePrice'].median():,.0f}")
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(df['SalePrice'], kde=True, ax=ax[0], color='steelblue')
        ax[0].set_title(f"SalePrice (skew {df['SalePrice'].skew():.2f})")
        sns.histplot(np.log1p(df['SalePrice']), kde=True, ax=ax[1], color='seagreen')
        ax[1].set_title(f"log1p(SalePrice) (skew {np.log1p(df['SalePrice']).skew():.2f})")
        st.pyplot(fig)
        st.caption("The price is right-skewed, so the models are trained on log1p(SalePrice).")
    with t2:
        stat = st.radio("Statistic", ["median", "mean"], horizontal=True)
        g = df.groupby('Neighborhood')['SalePrice'].agg([stat, 'count']).sort_values(stat)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(g.index, g[stat], color='steelblue'); ax.set_xlabel(f'{stat} SalePrice')
        st.pyplot(fig)
        groups = [x['SalePrice'].values for _, x in df.groupby('Neighborhood')]
        ssb = sum(len(x) * (x.mean() - df['SalePrice'].mean()) ** 2 for x in groups)
        eta = ssb / ((df['SalePrice'] - df['SalePrice'].mean()) ** 2).sum()
        st.info(f"η² (eta-squared) of Neighborhood = **{eta:.2f}**: the neighborhood alone explains about "
                f"{eta*100:.0f}% of the price variance.")
    with t3:
        feat = st.selectbox("Feature", ['GrLivArea', 'TotalBsmtSF', 'LotArea', 'YearBuilt', 'GarageArea',
                                        'OverallQual', 'GarageCars', 'FullBath', 'TotalSF', 'HouseAge'])
        fig, ax = plt.subplots(figsize=(9, 4.5))
        if df[feat].nunique() <= 12:
            sns.boxplot(data=df, x=feat, y='SalePrice', ax=ax)
        else:
            sns.scatterplot(data=df, x=feat, y='SalePrice', alpha=0.5, ax=ax)
        st.pyplot(fig)
        st.write(f"Correlation with SalePrice: **{df[feat].corr(df['SalePrice']):.2f}**")
    with t4:
        cols = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF',
                'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd', 'TotalSF', 'SalePrice']
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df[cols].corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
        st.pyplot(fig)

# ---------------------------------------------------------------- Results
elif page.startswith("🏆"):
    st.title("🏆 Model Results")
    st.write("All five models use their tuned hyperparameters from the notebook (Section 7). "
             "Validation and test sets were never used for tuning.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Test R²", f"{final_test['R²']:.3f}")
    c2.metric("Test RMSE", f"${final_test['RMSE ($)']:,.0f}")
    c3.metric("Test MAE", f"${final_test['MAE ($)']:,.0f}")
    split = st.radio("Split", ["Test", "Validation"], horizontal=True)
    t = res[res.Split == split].set_index('Model').drop(columns='Split').sort_values('R²', ascending=False)
    st.dataframe(t.style.format({'R²': '{:.3f}', 'RMSE ($)': '{:,.0f}', 'MAE ($)': '{:,.0f}'}))
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(t.index, t['RMSE ($)'], color='steelblue'); ax.set_ylabel('RMSE ($)')
    st.pyplot(fig)
    st.subheader("Predicted vs actual")
    mname = st.selectbox("Model", list(b.models), index=list(b.models).index(core.FINAL_MODEL))
    Xs, ys = (b.X_test, b.y_test) if split == "Test" else (b.X_val, b.y_val)
    pred = np.expm1(b.models[mname].predict(Xs)); act = np.expm1(ys)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].scatter(act, pred, alpha=0.6); lim = [act.min(), act.max()]
    ax[0].plot(lim, lim, 'r--'); ax[0].set_xlabel('Actual'); ax[0].set_ylabel('Predicted')
    ax[1].scatter(pred, act - pred, alpha=0.6); ax[1].axhline(0, color='r', ls='--')
    ax[1].set_xlabel('Predicted'); ax[1].set_ylabel('Residual ($)')
    st.pyplot(fig)

# ---------------------------------------------------------------- Importance
elif page.startswith("🔍"):
    st.title("🔍 Feature Importance")
    t1, t2 = st.tabs(["Permutation importance (final SVR)", "ElasticNet coefficients"])
    with t1:
        st.write("Shuffle one original column of the validation set and see how much the RMSE (log scale) "
                 "gets worse. Bigger = more important.")
        imp = get_permutation(b).head(15)
        fig, ax = plt.subplots(figsize=(8, 5))
        imp.sort_values().plot(kind='barh', color='steelblue', ax=ax); ax.set_xlabel('Increase in RMSE (log)')
        st.pyplot(fig)
        st.caption("Features that measure the same thing (e.g. TotalSF and GrLivArea) share their importance, "
                   "so a correlated feature can look less important than it really is.")
    with t2:
        co = core.elasticnet_coefs(b)
        st.write(f"ElasticNet keeps **{(co != 0).sum()} of {len(co)}** coefficients (the rest are exactly 0).")
        top = co.reindex(co.abs().sort_values(ascending=False).index).head(15).sort_values()
        fig, ax = plt.subplots(figsize=(8, 5))
        top.plot(kind='barh', color=['tomato' if v < 0 else 'steelblue' for v in top], ax=ax)
        st.pyplot(fig)

# ---------------------------------------------------------------- About
else:
    st.title("ℹ️ About the project")
    st.markdown(f"""
**Ames Housing Price Prediction**: a regression workflow on the Ames, Iowa housing data.

* **Data:** 1460 houses, 81 columns. 2 outliers removed (Id 524, 1299) → 1458 houses, 87 features after feature engineering.
* **Target:** `log1p(SalePrice)` (skew 1.88 → 0.12), predictions are converted back with `expm1`.
* **Split:** 70 / 15 / 15 (train / validation / test), stratified on price, `random_state=42`.
* **Pipelines:** median imputer + log1p / Yeo-Johnson + RobustScaler for numbers, one-hot for nominal,
  ordered ordinal encoding. All fitted on the training set only (no leakage).
* **Models:** SVR, KNN, Ridge, Lasso, ElasticNet (tuned with GridSearchCV, 5-fold CV) vs a median baseline.
* **Final model:** SVR, test R² = **{final_test['R²']:.3f}**, test RMSE = **${final_test['RMSE ($)']:,.0f}**,
  test MAE = **${final_test['MAE ($)']:,.0f}**. ElasticNet is the simpler alternative.

*Limits:* the model only knows Ames, Iowa between 2006 and 2010, so it should not be used for other cities or years.
""")
    if os.path.exists('Ames_Housing_Report.pdf'):
        with open('Ames_Housing_Report.pdf', 'rb') as f:
            st.download_button("📄 Download the project report (PDF)", f, file_name='Ames_Housing_Report.pdf')
