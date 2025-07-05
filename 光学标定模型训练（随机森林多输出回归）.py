import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt

# ----------------- 配置 matplotlib 中文与浅色配色 -----------------
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ----------------- 配置路径 -----------------
input_paths = {
    1998: r"D:\通用文件夹\定量遥感原理与应用\公共点取值结果\1998_pubilc_samples.xlsx",
    2003: r"D:\通用文件夹\定量遥感原理与应用\公共点取值结果\2003_pubilc_samples.xlsx",
    2008: r"D:\通用文件夹\定量遥感原理与应用\公共点取值结果\2008_pubilc_samples.xlsx",
    2013: r"D:\通用文件夹\定量遥感原理与应用\公共点取值结果\2013_pubilc_samples.xlsx",
    2018: r"D:\通用文件夹\定量遥感原理与应用\公共点取值结果\2018_pubilc_samples.xlsx",
}
output_base = r"D:\通用文件夹\定量遥感原理与应用\标定\方案二标定结果"
models_dir = os.path.join(output_base, "models")
metrics_dir = os.path.join(output_base, "metrics")
plots_dir = os.path.join(output_base, "plots")

os.makedirs(models_dir, exist_ok=True)
os.makedirs(metrics_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

# ----------------- 波段定义 -----------------
bands_TM = ['blue', 'green', 'red', 'NIR1', 'NIR2', 'MIR']
bands_OLI = ['costal_aerosol', 'blue', 'green', 'red', 'NIR1', 'SWIR1', 'SWIR2']
common_bands = ['blue', 'green', 'red', 'NIR1', 'SWIR1']

# ----------------- 1. 读取公共点样本 -----------------
data_raw = {}
for year, path in input_paths.items():
    if not os.path.exists(path):
        raise RuntimeError(f"样本文件不存在: {path}")
    df = pd.read_excel(path, engine='openpyxl')
    if 'point_id' not in df.columns:
        raise RuntimeError(f"{year} 文件缺少 'point_id' 列")
    data_raw[year] = df.copy()

# 检查字段完整性
for year, df in data_raw.items():
    expected = ['point_id'] + (bands_TM if year in [1998,2003,2008] else bands_OLI)
    missing = set(expected) - set(df.columns)
    if missing:
        raise RuntimeError(f"{year} 样本缺少列: {missing}")

# ----------------- 2. 内部同传感器归一化 using RandomForest -----------------
# 参考年
ref_TM = 2003
ref_OLI = 2013

internal_models = {'TM': None, 'OLI': None}
metrics_internal = []

# TM 内部：1998 & 2008 -> 2003
tm_X_list, tm_y_list = [], []
for year in [1998, 2008]:
    if year not in data_raw or ref_TM not in data_raw:
        continue
    df_sub = data_raw[year].set_index('point_id')
    df_ref = data_raw[ref_TM].set_index('point_id')
    common = df_sub.index.intersection(df_ref.index)
    if len(common)==0:
        continue
    tm_X_list.append(df_sub.loc[common, bands_TM].values)
    tm_y_list.append(df_ref.loc[common, bands_TM].values)

if tm_X_list:
    tm_X = np.vstack(tm_X_list)
    tm_y = np.vstack(tm_y_list)
    X_train_tm, X_test_tm, y_train_tm, y_test_tm = train_test_split(
        tm_X, tm_y, test_size=0.3, random_state=42
    )
    # 随机森林多输出回归：稳健、不易过拟合 :contentReference[oaicite:0]{index=0}
    rf_tm = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    )
    rf_tm.fit(X_train_tm, y_train_tm)
    # 评估并记录指标
    for i, band in enumerate(bands_TM):
        y_true = y_test_tm[:, i]
        y_pred = rf_tm.predict(X_test_tm)[:, i]
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        metrics_internal.append({'sensor':'TM', 'band':band, 'RMSE':rmse, 'R2':r2})
    # 保存模型
    joblib.dump(rf_tm, os.path.join(models_dir, "internal_TM_RF.pkl"))
    internal_models['TM'] = rf_tm

# OLI 内部：2018 -> 2013
oli_X_list, oli_y_list = [], []
for year in [2018]:
    if year not in data_raw or ref_OLI not in data_raw:
        continue
    df_sub = data_raw[year].set_index('point_id')
    df_ref = data_raw[ref_OLI].set_index('point_id')
    common = df_sub.index.intersection(df_ref.index)
    if len(common)==0:
        continue
    oli_X_list.append(df_sub.loc[common, bands_OLI].values)
    oli_y_list.append(df_ref.loc[common, bands_OLI].values)

if oli_X_list:
    oli_X = np.vstack(oli_X_list)
    oli_y = np.vstack(oli_y_list)
    X_train_oli, X_test_oli, y_train_oli, y_test_oli = train_test_split(
        oli_X, oli_y, test_size=0.3, random_state=42
    )
    rf_oli = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    )
    rf_oli.fit(X_train_oli, y_train_oli)
    for i, band in enumerate(bands_OLI):
        y_true = y_test_oli[:, i]
        y_pred = rf_oli.predict(X_test_oli)[:, i]
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        metrics_internal.append({'sensor':'OLI', 'band':band, 'RMSE':rmse, 'R2':r2})
    joblib.dump(rf_oli, os.path.join(models_dir, "internal_OLI_RF.pkl"))
    internal_models['OLI'] = rf_oli

# 保存内部归一化指标
df_metrics_internal = pd.DataFrame(metrics_internal)
df_metrics_internal.to_csv(
    os.path.join(metrics_dir, "内部归一化_RF指标.csv"),
    index=False, encoding='utf-8-sig'
)

# ----------------- 3. 跨传感器映射 TM->OLI -----------------
metrics_cross = []
cross_model = None
if ref_TM in data_raw and ref_OLI in data_raw:
    df_TM_ref = data_raw[ref_TM].set_index('point_id')
    df_OLI_ref = data_raw[ref_OLI].set_index('point_id')
    common = df_TM_ref.index.intersection(df_OLI_ref.index)
    if len(common)>0:
        X_vals = df_TM_ref.loc[common, bands_TM].values
        y_vals = df_OLI_ref.loc[common, common_bands].values
        X_train_cross, X_test_cross, y_train_cross, y_test_cross = train_test_split(
            X_vals, y_vals, test_size=0.3, random_state=42
        )
        rf_cross = MultiOutputRegressor(
            RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        )
        rf_cross.fit(X_train_cross, y_train_cross)
        for i, band in enumerate(common_bands):
            y_true = y_test_cross[:, i]
            y_pred = rf_cross.predict(X_test_cross)[:, i]
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            metrics_cross.append({'band':band, 'RMSE':rmse, 'R2':r2})
        joblib.dump(rf_cross, os.path.join(models_dir, "cross_TM2OLI_RF.pkl"))
        cross_model = rf_cross

# 保存跨传感器指标
df_metrics_cross = pd.DataFrame(metrics_cross)
df_metrics_cross.to_csv(
    os.path.join(metrics_dir, "跨传感器_RF指标.csv"),
    index=False, encoding='utf-8-sig'
)

# ----------------- 4. 性能可视化，多图合并子图保存 -----------------
# 以下示例绘制内部归一化和跨传感器映射的多种可视化，使用浅色配色、合并子图：
# 1) 真实 vs 预测 散点 + 1:1 参考线
# 2) 残差直方图
# 3) 残差箱线图（各波段误差分布）
# 4) 特征重要性条形图（对于内部，用各波段输入对输出每波段的重要性；对于跨，用 TM 输入对每个公共波段的重要性）

# 定义浅色配色
colors = {
    'scatter': '#a6cee3',     # 浅蓝
    'line':   '#1f78b4',      # 深蓝
    'hist':   '#b2df8a',      # 浅绿
    'box':    '#fb9a99',      # 浅红
    'bar':    '#fdbf6f'       # 浅橙
}

# 4.1 内部归一化可视化：TM
if internal_models['TM'] is not None and tm_X_list:
    # 使用之前的测试集 X_test_tm, y_test_tm
    y_pred_tm = internal_models['TM'].predict(X_test_tm)  # shape (n_samples, len(bands_TM))
    # 4.1.1 真实 vs 预测：多个子图
    n = len(bands_TM)
    cols = 3
    rows = int(np.ceil(n/cols))
    fig = plt.figure(figsize=(cols*4, rows*4))
    for i, band in enumerate(bands_TM):
        ax = fig.add_subplot(rows, cols, i+1)
        ax.scatter(y_test_tm[:, i], y_pred_tm[:, i], c=colors['scatter'], s=10, alpha=0.6)
        lims = [min(y_test_tm[:, i].min(), y_pred_tm[:, i].min()),
                max(y_test_tm[:, i].max(), y_pred_tm[:, i].max())]
        ax.plot(lims, lims, color=colors['line'], linewidth=1)
        ax.set_title(f"TM 内部归一化: {band} 真 vs 预测")
        ax.set_xlabel("真实值")
        ax.set_ylabel("预测值")
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "TM_internal_scatter_true_vs_pred.png"), dpi=300)
    plt.close(fig)

    # 4.1.2 残差直方图：多个子图
    fig = plt.figure(figsize=(cols*4, rows*3))
    for i, band in enumerate(bands_TM):
        ax = fig.add_subplot(rows, cols, i+1)
        resid = y_test_tm[:, i] - y_pred_tm[:, i]
        ax.hist(resid, bins=30, color=colors['hist'], alpha=0.7)
        ax.set_title(f"TM 内部归一化残差直方图: {band}")
        ax.set_xlabel("残差")
        ax.set_ylabel("频数")
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "TM_internal_residual_histograms.png"), dpi=300)
    plt.close(fig)

    # 4.1.3 残差箱线图：合并各波段残差
    resid_matrix = [y_test_tm[:, i] - y_pred_tm[:, i] for i in range(len(bands_TM))]
    fig, ax = plt.subplots(figsize=(len(bands_TM)*1.5, 5))
    ax.boxplot(resid_matrix, labels=bands_TM, patch_artist=True,
               boxprops=dict(facecolor=colors['box'], alpha=0.6))
    ax.set_title("TM 内部归一化残差箱线图")
    ax.set_ylabel("残差")
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "TM_internal_residual_boxplot.png"), dpi=300)
    plt.close(fig)

    # 4.1.4 特征重要性：MultiOutputRegressor 内部，每个输出波段对应一个 estimator
    # 提取各输出模型的 feature_importances_
    rf_tm_estims = internal_models['TM'].estimators_
    # 对于每个输出波段，绘制输入特征重要性
    fig = plt.figure(figsize=(cols*4, rows*3))
    for i, band in enumerate(bands_TM):
        ax = fig.add_subplot(rows, cols, i+1)
        importances = rf_tm_estims[i].feature_importances_
        ax.bar(bands_TM, importances, color=colors['bar'], alpha=0.7)
        ax.set_title(f"TM 内部归一化 特征重要性 for {band}")
        ax.set_xticklabels(bands_TM, rotation=45, ha='right')
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "TM_internal_feature_importance.png"), dpi=300)
    plt.close(fig)

# 4.2 内部归一化可视化：OLI
if internal_models['OLI'] is not None and oli_X_list:
    y_pred_oli = internal_models['OLI'].predict(X_test_oli)
    n = len(bands_OLI)
    cols = 4
    rows = int(np.ceil(n/cols))
    # 真 vs 预测散点
    fig = plt.figure(figsize=(cols*4, rows*4))
    for i, band in enumerate(bands_OLI):
        ax = fig.add_subplot(rows, cols, i+1)
        ax.scatter(y_test_oli[:, i], y_pred_oli[:, i], c=colors['scatter'], s=10, alpha=0.6)
        lims = [min(y_test_oli[:, i].min(), y_pred_oli[:, i].min()),
                max(y_test_oli[:, i].max(), y_pred_oli[:, i].max())]
        ax.plot(lims, lims, color=colors['line'], linewidth=1)
        ax.set_title(f"OLI 内部归一化: {band} 真 vs 预测")
        ax.set_xlabel("真实值")
        ax.set_ylabel("预测值")
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "OLI_internal_scatter_true_vs_pred.png"), dpi=300)
    plt.close(fig)

    # 残差直方图
    fig = plt.figure(figsize=(cols*4, rows*3))
    for i, band in enumerate(bands_OLI):
        ax = fig.add_subplot(rows, cols, i+1)
        resid = y_test_oli[:, i] - y_pred_oli[:, i]
        ax.hist(resid, bins=30, color=colors['hist'], alpha=0.7)
        ax.set_title(f"OLI 内部归一化残差: {band}")
        ax.set_xlabel("残差")
        ax.set_ylabel("频数")
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "OLI_internal_residual_histograms.png"), dpi=300)
    plt.close(fig)

    # 残差箱线图
    resid_matrix = [y_test_oli[:, i] - y_pred_oli[:, i] for i in range(len(bands_OLI))]
    fig, ax = plt.subplots(figsize=(len(bands_OLI)*1.2, 5))
    ax.boxplot(resid_matrix, labels=bands_OLI, patch_artist=True,
               boxprops=dict(facecolor=colors['box'], alpha=0.6))
    ax.set_title("OLI 内部归一化残差箱线图")
    ax.set_ylabel("残差")
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "OLI_internal_residual_boxplot.png"), dpi=300)
    plt.close(fig)

    # 特征重要性
    rf_oli_estims = internal_models['OLI'].estimators_
    fig = plt.figure(figsize=(cols*4, rows*3))
    for i, band in enumerate(bands_OLI):
        ax = fig.add_subplot(rows, cols, i+1)
        importances = rf_oli_estims[i].feature_importances_
        ax.bar(bands_OLI, importances, color=colors['bar'], alpha=0.7)
        ax.set_title(f"OLI 内部归一化 特征重要性 for {band}")
        ax.set_xticklabels(bands_OLI, rotation=45, ha='right')
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "OLI_internal_feature_importance.png"), dpi=300)
    plt.close(fig)

# 4.3 跨传感器映射可视化 TM->OLI
if cross_model is not None:
    y_pred_cross = cross_model.predict(X_test_cross)  # shape (n_samples, len(common_bands))
    n = len(common_bands)
    cols = 3
    rows = int(np.ceil(n/cols))
    # 真 vs 预测散点
    fig = plt.figure(figsize=(cols*4, rows*4))
    for i, band in enumerate(common_bands):
        ax = fig.add_subplot(rows, cols, i+1)
        ax.scatter(y_test_cross[:, i], y_pred_cross[:, i],
                   c=colors['scatter'], s=10, alpha=0.6)
        lims = [min(y_test_cross[:, i].min(), y_pred_cross[:, i].min()),
                max(y_test_cross[:, i].max(), y_pred_cross[:, i].max())]
        ax.plot(lims, lims, color=colors['line'], linewidth=1)
        ax.set_title(f"跨传感器映射: {band} 真 vs 预测")
        ax.set_xlabel("真实值")
        ax.set_ylabel("预测值")
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "cross_scatter_true_vs_pred.png"), dpi=300)
    plt.close(fig)

    # 残差直方图
    fig = plt.figure(figsize=(cols*4, rows*3))
    for i, band in enumerate(common_bands):
        ax = fig.add_subplot(rows, cols, i+1)
        resid = y_test_cross[:, i] - y_pred_cross[:, i]
        ax.hist(resid, bins=30, color=colors['hist'], alpha=0.7)
        ax.set_title(f"跨传感器残差: {band}")
        ax.set_xlabel("残差")
        ax.set_ylabel("频数")
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "cross_residual_histograms.png"), dpi=300)
    plt.close(fig)

    # 残差箱线图
    resid_matrix = [y_test_cross[:, i] - y_pred_cross[:, i] for i in range(len(common_bands))]
    fig, ax = plt.subplots(figsize=(len(common_bands)*1.5, 5))
    ax.boxplot(resid_matrix, labels=common_bands, patch_artist=True,
               boxprops=dict(facecolor=colors['box'], alpha=0.6))
    ax.set_title("跨传感器归一化残差箱线图")
    ax.set_ylabel("残差")
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "cross_residual_boxplot.png"), dpi=300)
    plt.close(fig)

    # 特征重要性：MultiOutputRegressor 包含多个 estimator
    rf_cross_estims = cross_model.estimators_
    fig = plt.figure(figsize=(cols*4, rows*3))
    for i, band in enumerate(common_bands):
        ax = fig.add_subplot(rows, cols, i+1)
        importances = rf_cross_estims[i].feature_importances_
        ax.bar(bands_TM, importances, color=colors['bar'], alpha=0.7)
        ax.set_title(f"跨传感器 特征重要性 for {band}")
        ax.set_xticklabels(bands_TM, rotation=45, ha='right')
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "cross_feature_importance.png"), dpi=300)
    plt.close(fig)

print("随机森林模型训练完成，性能可视化图已保存至:", plots_dir)
print("模型文件保存在:", models_dir)
print("指标 CSV 保存在:", metrics_dir)
