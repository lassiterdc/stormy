#%% load libraries
import pandas as pd
import numpy as np
from tqdm import tqdm
from _inputs import *
import geopandas as gpd
import cartopy.crs as ccrs
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xarray as xr

#%% load data
df_variable_nodes, ds_flood_attribution, ds_sst_compound, ds_sst_freebndry, ds_events, _, _, _, _ = return_attribution_data()
#%%
allvars = predictors + response
lst_sse_hat_per_node = []
lst_s_rf_parameters = []
lst_reg_rf_models = []
df_rf_perf = pd.DataFrame()
count = -1
node_count = -1
for node in tqdm(df_variable_nodes.node.values):
    node_count += 1
    ds_flood_attribution_sub = ds_flood_attribution.sel(node_id = node)
    ds_sst_compound_sub = ds_sst_compound.sel(node_id = node)
    # make sure this node has flooded in the simulation
    if float(ds_sst_compound_sub.node_flooding_cubic_meters.sum()) <= 0:
        continue
    # if there is no variance in flood attribution, pass
    if np.nanstd(ds_flood_attribution_sub.flood_attribution.values) == 0:
        continue

    # only use rows where flood occured
    ds_sst_compound_sub = ds_sst_compound_sub.where(ds_sst_compound_sub.node_flooding_cubic_meters>0, drop = True)
    # create 2D dataframes with the data I want to use
    ds_merged = xr.merge([ds_events, ds_flood_attribution_sub, ds_sst_compound_sub], join = "right")
    df_merged = ds_merged.to_dataframe().reset_index()
    df_merged = df_merged.dropna()


    # normalize variables
    df_merged_subset = df_merged.loc[:, allvars]
    
    preds = df_merged.loc[:, predictors]
    
    resp = df_merged[response]
    scaler = preprocessing.StandardScaler().fit(preds)
    preds_scaled = scaler.transform(preds)
    df_preds_scaled = pd.DataFrame(preds_scaled)
    df_preds_scaled.columns = preds.columns

    # lst_sse_hat_per_node.append()
    if len(resp) < min_rows_for_fitting:
        continue


    for ntree in ar_trees:
        for depth in ar_depths:
            count += 1
            df_rf_perf.loc[count, "node"] = node
            df_rf_perf.loc[count, "trees"] = ntree
            df_rf_perf.loc[count, "depth"] = depth
            lst_sse_hat = []
            for test_id in np.arange(num_fits_for_estimating_sse):
                X_train, X_test, y_train, y_test = train_test_split(df_preds_scaled, resp, test_size=0.1)
                regr = RandomForestRegressor(n_estimators = ntree, max_depth = depth)
                regr.fit(X_train, y_train.values.ravel())
                y_train_pred = regr.predict(X_train)
                y_test_pred = regr.predict(X_test)
                sse_fit = mean_squared_error(y_train, y_train_pred)
                sse_hat = mean_squared_error(y_test, y_test_pred)
                lst_sse_hat.append(sse_hat)
            df_rf_perf.loc[count, "mean_sse_hat"] = float(np.mean(lst_sse_hat))
            df_rf_perf.loc[count, "var_sse_hat"] = float(np.std(lst_sse_hat))
    # choosing an architecture
    df_rf_perf_node = df_rf_perf[(df_rf_perf.node == node)]
    min_acceptable_sse = df_rf_perf_node.mean_sse_hat.quantile(q = 0.1, interpolation = "nearest")
    # filter only those with acceptable mean SSE's
    df_rf_perf_node = df_rf_perf_node[df_rf_perf_node.mean_sse_hat <= min_acceptable_sse]
    # choose the one with the lowest variance
    s_node_rf_parameters = df_rf_perf_node.loc[df_rf_perf_node.var_sse_hat.idxmin(), :]
    lst_s_rf_parameters.append(s_node_rf_parameters)

df_rf_parameters = pd.DataFrame(lst_s_rf_parameters)
df_rf_perf.to_csv("analysis/random_forest/g_random_forest_parameter_tuning.csv")
df_rf_parameters.to_csv("analysis/random_forest/g_random_forest_parameters.csv")