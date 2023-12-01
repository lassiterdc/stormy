#%% load libs
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from _inputs import *
from _inputs import return_attribution_data
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor

#%% load data
df_rf_perf = pd.read_csv("analysis/random_forest/g_random_forest_parameter_tuning.csv")
df_rf_parameters = pd.read_csv("analysis/random_forest/g_random_forest_parameters.csv")

_, ds_flood_attribution, ds_sst_compound, _, ds_events, _, _, _, _ = return_attribution_data()

allvars = predictors + response
#%% analyzing
lst_fit_models = []
lst_df_importances = []
for idx, row in df_rf_parameters.iterrows():
    node = row.node
    ds_flood_attribution_sub = ds_flood_attribution.sel(node_id = node)
    ds_sst_compound_sub = ds_sst_compound.sel(node_id = node)

    # only use rows where flood occureds
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

    regr = RandomForestRegressor(n_estimators = int(row.trees), max_depth = int(row.depth))
    regr.fit(df_preds_scaled, resp.values.ravel())
    lst_fit_models.append(regr)

    df_importance_node = pd.DataFrame(regr.feature_importances_).T
    df_importance_node.columns = preds.columns
    lst_df_importances.append(df_importance_node)

df_importance = pd.concat(lst_df_importances)
df_importance["node"] = df_rf_parameters.node.values
df_importance = df_importance.set_index("node")


#%% iterate over columns and map the nodes with the highest importance scores
#for each row, rank importance form 1-5
df_importance_ranked = df_importance.rank(axis = 1, ascending = False).astype(int)


lst_val_counts = []
for column in df_importance_ranked:
    s = df_importance_ranked[column].value_counts()
    lst_val_counts.append(s)
    # fig, ax = plt.subplots()
    # df_importance_ranked[column].value_counts().plot(kind = 'barh', ax = ax)
    # ax.set_xlabel("count")
    # ax.set_ylabel("feature importance ranking")
    # ax.set_title(column)
df_val_counts = pd.DataFrame(lst_val_counts).fillna(0)
df_val_counts = df_val_counts.T.sort_index().T

fig, ax = plt.subplots(dpi = 300, figsize = (8, 4))

lst_colors = ['#d53e4f','#fc8d59','#fee08b','#e6f598','#99d594','#3288bd']
lst_colors.reverse()

df_val_counts.plot.barh(ax = ax, color = lst_colors, label = "importance ranking", legend = False, stacked = True)
ax.set_xlabel("node_count")
# ax.set_ylabel("count")
ax.set_title("Random Forest Feature Importance Plot for Predicting Flood Attribution")
fig.legend(bbox_to_anchor=(1.08,.92))
plt.tight_layout()
# handles, labels = ax.get_legend_handles_labels()
plt.yticks(rotation=45)
plt.savefig("analysis/plots/f_importance.png")

#%% show nodes with different important features
df_importance_ranked[df_importance_ranked.max_mm_per_hour == 1]