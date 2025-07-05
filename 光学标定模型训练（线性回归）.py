import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 避免中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ---------- 配置路径 ----------
input_paths = {
    1998: r"D:\通用文件夹\定量遥感原理与应用\公共点取值结果\1998_pubilc_samples.xlsx",
    2003: r"D:\通用文件夹\定量遥感原理与应用\公共点取值结果\2003_pubilc_samples.xlsx",
    2008: r"D:\通用文件夹\定量遥感原理与应用\公共点取值结果\2008_pubilc_samples.xlsx",
    2013: r"D:\通用文件夹\定量遥感原理与应用\公共点取值结果\2013_pubilc_samples.xlsx",
    2018: r"D:\通用文件夹\定量遥感原理与应用\公共点取值结果\2018_pubilc_samples.xlsx",
}
output_base = r"D:\通用文件夹\定量遥感原理与应用\标定\方案一标定结果"
output_dir = os.path.join(output_base, "转换结果")
plots_dir = os.path.join(output_base, "plots")
metrics_dir = os.path.join(output_base, "metrics")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(metrics_dir, exist_ok=True)

# ---------- 波段定义 ----------
bands_TM = ['blue','green','red','NIR1','NIR2','MIR']
bands_OLI = ['costal_aerosol','blue','green','red','NIR1','SWIR1','SWIR2']
common_bands = ['blue','green','red','NIR1','SWIR1']  # 最终统一频段，SWIR1 统一

# ---------- 1. 读取数据 ----------
data_raw = {}
for year, path in input_paths.items():
    if os.path.exists(path):
        df = pd.read_excel(path, engine='openpyxl')
        if 'point_id' not in df.columns:
            print(f"警告: {year} 文件缺少 'point_id' 列，已跳过")
            continue
        data_raw[year] = df.copy()
    else:
        print(f"警告: 文件不存在: {path}，已跳过年份 {year}")

# 检查列名并过滤不可用年份
valid_years = []
for year, df in list(data_raw.items()):
    expected = ['point_id'] + (bands_TM if year in [1998,2003,2008] else bands_OLI)
    missing = set(expected) - set(df.columns)
    if missing:
        print(f"警告: {year} 文件缺少列: {missing}，移除该年份")
        data_raw.pop(year)
    else:
        valid_years.append(year)
valid_years.sort()

if not valid_years:
    print("无有效年份数据，退出.")
else:
    # ---------- 2. 同传感器内部归一化 ----------
    ref_TM = 2003
    years_TM = [y for y in [1998,2003,2008] if y in valid_years]
    ref_OLI = 2013
    years_OLI = [y for y in [2013,2018] if y in valid_years]

    data_norm = {}
    metrics_internal = []  # 存储内部归一化指标: sensor, year, band, RMSE_before, RMSE_after, R2_after

    # TM 内部归一化到 2003
    if ref_TM in years_TM:
        for year in years_TM:
            df = data_raw[year].copy()
            if year == ref_TM:
                data_norm[year] = df.copy()
            else:
                df_ref = data_raw[ref_TM].set_index('point_id')
                df_sub = df.set_index('point_id')
                common = df_ref.index.intersection(df_sub.index)
                df_ref_c = df_ref.loc[common]
                df_sub_c = df_sub.loc[common]
                df_norm = df_sub.copy()
                for band in bands_TM:
                    y_true = df_ref_c[band].values.flatten()
                    y_pred_before = df_sub_c[band].values.flatten()
                    mse_before = mean_squared_error(y_true, y_pred_before)
                    X = df_sub_c[[band]].values
                    y = df_ref_c[[band]].values
                    model = LinearRegression()
                    model.fit(X, y)
                    a, b = model.coef_[0,0], model.intercept_[0]
                    y_pred_after = (df_sub_c[band].values * a + b).flatten()
                    mse_after = mean_squared_error(y_true, y_pred_after)
                    r2 = r2_score(y_true, y_pred_after)
                    metrics_internal.append({
                        'sensor': 'TM', 'year': year, 'band': band,
                        'RMSE_before': np.sqrt(mse_before), 'RMSE_after': np.sqrt(mse_after), 'R2_after': r2
                    })
                    df_norm[band] = df_sub[band] * a + b
                data_norm[year] = df_norm.reset_index()
    else:
        print("无 TM 参考年，跳过 TM 内部归一化")

    # OLI 内部归一化到 2013
    if ref_OLI in years_OLI:
        for year in years_OLI:
            df = data_raw[year].copy()
            if year == ref_OLI:
                data_norm[year] = df.copy()
            else:
                df_ref = data_raw[ref_OLI].set_index('point_id')
                df_sub = df.set_index('point_id')
                common = df_ref.index.intersection(df_sub.index)
                df_ref_c = df_ref.loc[common]
                df_sub_c = df_sub.loc[common]
                df_norm = df_sub.copy()
                for band in bands_OLI:
                    y_true = df_ref_c[band].values.flatten()
                    y_pred_before = df_sub_c[band].values.flatten()
                    mse_before = mean_squared_error(y_true, y_pred_before)
                    X = df_sub_c[[band]].values
                    y = df_ref_c[[band]].values
                    model = LinearRegression()
                    model.fit(X, y)
                    a, b = model.coef_[0,0], model.intercept_[0]
                    y_pred_after = (df_sub_c[band].values * a + b).flatten()
                    mse_after = mean_squared_error(y_true, y_pred_after)
                    r2 = r2_score(y_true, y_pred_after)
                    metrics_internal.append({
                        'sensor': 'OLI', 'year': year, 'band': band,
                        'RMSE_before': np.sqrt(mse_before), 'RMSE_after': np.sqrt(mse_after), 'R2_after': r2
                    })
                    df_norm[band] = df_sub[band] * a + b
                data_norm[year] = df_norm.reset_index()
    else:
        print("无 OLI 参考年，跳过 OLI 内部归一化")

    # 保存内部归一化指标
    if metrics_internal:
        df_metrics_internal = pd.DataFrame(metrics_internal)
        path_metrics_internal = os.path.join(metrics_dir, "内部归一化指标.csv")
        df_metrics_internal.to_csv(path_metrics_internal, index=False, encoding='utf-8-sig')

        # 绘制内部归一化前后 RMSE 对比（每个波段一个子图，横轴年份，离散点）
        for sensor in df_metrics_internal['sensor'].unique():
            df_s = df_metrics_internal[df_metrics_internal['sensor'] == sensor]
            bands = df_s['band'].unique()
            n = len(bands)
            cols = 3
            rows = (n + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*3))
            axes = axes.flatten()
            for i, band in enumerate(bands):
                ax = axes[i]
                df_b = df_s[df_s['band'] == band]
                years = df_b['year'].tolist()
                before = df_b['RMSE_before'].values
                after = df_b['RMSE_after'].values
                ax.bar([y-0.2 for y in years], before, width=0.4, label='归一化前RMSE')
                ax.bar([y+0.2 for y in years], after, width=0.4, label='归一化后RMSE')
                ax.set_xticks(years)
                ax.set_xticklabels(years)
                ax.set_title(f"{sensor} {band} 年份归一化效果")
                ax.set_xlabel("年份")
                ax.set_ylabel("RMSE")
                ax.legend()
            # 隐藏多余子图
            for j in range(n, len(axes)):
                fig.delaxes(axes[j])
            plt.tight_layout()
            fname = os.path.join(plots_dir, f"{sensor}_内部归一化_RMSE子图.png")
            plt.savefig(fname)
            plt.close()
            print(f"已保存：{fname} —— 含义：此图展示同传感器各波段在不同年份归一化前后RMSE对比，横轴为年份离散点，蓝柱为归一化前误差，橙柱为归一化后误差，可观察各波段归一化效果优劣，关注RMSE下降程度。")

    # ---------- 3. 跨传感器映射至统一 SWIR1 ----------
    if ref_TM in data_norm and ref_OLI in data_norm:
        df_TM_ref = data_norm[ref_TM].set_index('point_id')
        df_OLI_ref = data_norm[ref_OLI].set_index('point_id')
        common_pts = df_TM_ref.index.intersection(df_OLI_ref.index)
        df_TM_ref_c = df_TM_ref.loc[common_pts]
        df_OLI_ref_c = df_OLI_ref.loc[common_pts]

        models = {}
        metrics_cross = []
        for band in ['blue','green','red','NIR1']:
            X = df_TM_ref_c[[band]].values
            y = df_OLI_ref_c[[band]].values
            model = LinearRegression()
            model.fit(X, y)
            a, b = model.coef_[0,0], model.intercept_[0]
            models[band] = (a, b)
            y_true = df_OLI_ref_c[band].values.flatten()
            y_pred = (df_TM_ref_c[band].values * a + b).flatten()
            metrics_cross.append({'band': band, 'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)), 'R2': r2_score(y_true, y_pred)})
        # MIR->SWIR1
        if 'MIR' in df_TM_ref_c.columns and 'SWIR1' in df_OLI_ref_c.columns:
            X_mir = df_TM_ref_c[['MIR']].values
            y_swir1 = df_OLI_ref_c[['SWIR1']].values
            model_mir = LinearRegression()
            model_mir.fit(X_mir, y_swir1)
            a_m, b_m = model_mir.coef_[0,0], model_mir.intercept_[0]
            models['MIR2SWIR1'] = (a_m, b_m)
            y_true = df_OLI_ref_c['SWIR1'].values.flatten()
            y_pred = (df_TM_ref_c['MIR'].values * a_m + b_m).flatten()
            metrics_cross.append({'band': 'SWIR1', 'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)), 'R2': r2_score(y_true, y_pred)})

        df_metrics_cross = pd.DataFrame(metrics_cross)
        df_metrics_cross.to_csv(os.path.join(metrics_dir, "跨传感器映射指标.csv"), index=False, encoding='utf-8-sig')

        # 绘制跨传感器映射指标（RMSE和R2）子图：每波段两个柱
        bands = df_metrics_cross['band'].tolist()
        x = np.arange(len(bands))
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax2 = ax1.twinx()
        ax1.bar(x-0.2, df_metrics_cross['RMSE'], width=0.4, color='skyblue', label='RMSE')
        ax2.bar(x+0.2, df_metrics_cross['R2'], width=0.4, color='orange', label='R2')
        ax1.set_xticks(x)
        ax1.set_xticklabels(bands)
        ax1.set_xlabel("波段")
        ax1.set_ylabel("RMSE", color='skyblue')
        ax2.set_ylabel("R²", color='orange')
        plt.title("跨传感器映射性能指标")
        # 合并图例
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines+lines2, labels+labels2, loc='upper right')
        plt.tight_layout()
        fname = os.path.join(plots_dir, "跨传感器映射指标_RMSE_R2.png")
        plt.savefig(fname)
        plt.close()
        print(f"已保存：{fname} —— 含义：此图展示跨传感器不同波段映射后的性能指标，左Y轴RMSE（误差，越低越好），右Y轴R²（拟合优度，越接近1越好），横轴为波段名称，可直观比较各波段映射效果。")

        # ---------- 4. 应用到所有年份，生成统一频段结果 ----------
        for year, df in data_norm.items():
            df_out = pd.DataFrame({'point_id': df['point_id']})
            if year in years_TM:
                for band in ['blue','green','red','NIR1']:
                    a, b = models[band]
                    df_out[band] = df[band] * a + b
                a_m, b_m = models.get('MIR2SWIR1', (np.nan, np.nan))
                df_out['SWIR1'] = df['MIR'] * a_m + b_m
            else:
                for band in common_bands:
                    df_out[band] = df[band]
            out_path = os.path.join(output_dir, f"{year}_统一频段.xlsx")
            df_out.to_excel(out_path, index=False)
    else:
        print("缺少归一化后的参考年，无法跨传感器映射")

    # ---------- 5. 时序一致性全局展示 ----------
    # 选若干示例点，绘制所有年份统一频段时序曲线，带离散年份刻度
    if 'common_pts' in locals() and len(common_pts) > 0:
        example_pts = list(common_pts)[:3]
        yrs = sorted(valid_years)
        # 蓝色通道和SWIR1
        for band in ['blue', 'SWIR1']:
            fig, ax = plt.subplots(figsize=(8, 5))
            for pt in example_pts:
                vals = []
                for y in yrs:
                    df_uni = pd.read_excel(os.path.join(output_dir, f"{y}_统一频段.xlsx"), engine='openpyxl').set_index('point_id')
                    val = df_uni.loc[pt, band] if pt in df_uni.index and band in df_uni.columns else np.nan
                    vals.append(val)
                ax.plot(yrs, vals, marker='o', label=f"点 {pt}")
            ax.set_xticks(yrs)
            ax.set_xlabel("年份 (离散)")
            ax.set_ylabel("反射率")
            ax.set_title(f"{band} 通道统一频段时序示例")
            ax.legend()
            plt.tight_layout()
            fname = os.path.join(plots_dir, f"示例点_{band}_统一时序.png")
            plt.savefig(fname)
            plt.close()
            print(f"已保存：{fname} —— 含义：此图展示若干公共点在各年份统一频段下的反射率时序，横轴为离散年份，纵轴为反射率值。理想情况下应平稳无明显跳变，可用以评估时序一致性。")
    else:
        print("无公共点或无法获取，跳过时序一致性图")
