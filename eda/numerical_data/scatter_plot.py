fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(
    nrows=3, ncols=2, figsize=(16, 13))
OverallQual_scatter_plot = pd.concat(
    [df_train['SalePrice'], df_train['OverallQual']], axis=1)
sns.regplot(x='OverallQual', y='SalePrice',
            data=OverallQual_scatter_plot, scatter=True, fit_reg=True, ax=ax1)
TotalBsmtSF_scatter_plot = pd.concat(
    [df_train['SalePrice'], df_train['TotalBsmtSF']], axis=1)
sns.regplot(x='TotalBsmtSF', y='SalePrice',
            data=TotalBsmtSF_scatter_plot, scatter=True, fit_reg=True, ax=ax2)
GrLivArea_scatter_plot = pd.concat(
    [df_train['SalePrice'], df_train['GrLivArea']], axis=1)
sns.regplot(x='GrLivArea', y='SalePrice',
            data=GrLivArea_scatter_plot, scatter=True, fit_reg=True, ax=ax3)
GarageCars_scatter_plot = pd.concat(
    [df_train['SalePrice'], df_train['GarageCars']], axis=1)
sns.regplot(x='GarageCars', y='SalePrice',
            data=GarageCars_scatter_plot, scatter=True, fit_reg=True, ax=ax4)
FullBath_scatter_plot = pd.concat(
    [df_train['SalePrice'], df_train['FullBath']], axis=1)
sns.regplot(x='FullBath', y='SalePrice', data=FullBath_scatter_plot,
            scatter=True, fit_reg=True, ax=ax5)
YearBuilt_scatter_plot = pd.concat(
    [df_train['SalePrice'], df_train['YearBuilt']], axis=1)
sns.regplot(x='YearBuilt', y='SalePrice',
            data=YearBuilt_scatter_plot, scatter=True, fit_reg=True, ax=ax6)
YearRemodAdd_scatter_plot = pd.concat(
    [df_train['SalePrice'], df_train['YearRemodAdd']], axis=1)
YearRemodAdd_scatter_plot.plot.scatter('YearRemodAdd', 'SalePrice')

# Target Feature "SalePrice"와 가장 밀접한 연관이 있다고 판단됐던 변수들의 Scatter Plot을 그립니다.
# OverallQual, GarageCars, Fullbath와 같은 변수들은 실제로는 범주형 데이터의 특징을 보인다고 할 수 있습니다. (등급, 갯수 등을 의미하기 때문)
