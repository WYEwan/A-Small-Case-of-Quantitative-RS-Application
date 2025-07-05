import os
import pandas as pd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from sklearn.linear_model import LinearRegression

# ---------- 配置路径 ----------
# 公共点样本路径
input_paths = {
    1998: r"1998_pubilc_samples.xlsx",
    2003: r"2003_pubilc_samples.xlsx",
    2008: r"2008_pubilc_samples.xlsx",
    2013: r"2013_pubilc_samples.xlsx",
    2018: r"2018_pubilc_samples.xlsx",
}
# 遥感影像路径
image_paths = {
    1998: r"1998.tif",
    2003: r"2003.tif",
    2008: r"2008.tif",
    2013: r"2013.tif",
    2018: r"2018.tif",
}
# 输出文件夹
output_dir = r"D:\通用文件夹"
os.makedirs(output_dir, exist_ok=True)

# ---------- 波段定义 ----------
bands_TM = ['blue', 'green', 'red', 'NIR1', 'NIR2', 'MIR']
bands_OLI = ['costal_aerosol', 'blue', 'green', 'red', 'NIR1', 'SWIR1', 'SWIR2']
common_bands = ['blue', 'green', 'red', 'NIR1', 'SWIR1']  # 统一输出波段

# ---------- 1. 读取公共点样本并计算校正模型 ----------
# 读取样本
data_raw = {}
for year, path in input_paths.items():
    if os.path.exists(path):
        df = pd.read_excel(path, engine='openpyxl')
        if 'point_id' not in df.columns:
            raise RuntimeError(f"{year} 样本缺少 'point_id' 列")
        data_raw[year] = df.copy()
    else:
        raise RuntimeError(f"样本文件不存在: {path}")

# 检查列
for year, df in data_raw.items():
    expected = ['point_id'] + (bands_TM if year in [1998,2003,2008] else bands_OLI)
    missing = set(expected) - set(df.columns)
    if missing:
        raise RuntimeError(f"{year} 样本缺少列: {missing}")

# 内部归一化模型存储: {year: {band: (a,b)}} 不含参考年自身
internal_models = {'TM': {}, 'OLI': {}}
ref_TM = 2003
ref_OLI = 2013

# Landsat5 TM 内部归一化模型: 拟合 ref_TM = a * sub + b
for year in [1998, 2008]:
    df_ref = data_raw[ref_TM].set_index('point_id')
    df_sub = data_raw[year].set_index('point_id')
    common = df_ref.index.intersection(df_sub.index)
    df_ref_c = df_ref.loc[common]
    df_sub_c = df_sub.loc[common]
    models = {}
    for band in bands_TM:
        X = df_sub_c[[band]].values
        y = df_ref_c[[band]].values
        model = LinearRegression().fit(X, y)
        a, b = model.coef_[0,0], model.intercept_[0]
        models[band] = (a, b)
    internal_models['TM'][year] = models
# Landsat8 OLI 内部归一化模型: 拟合 ref_OLI = a * sub + b
for year in [2018]:
    df_ref = data_raw[ref_OLI].set_index('point_id')
    df_sub = data_raw[year].set_index('point_id')
    common = df_ref.index.intersection(df_sub.index)
    df_ref_c = df_ref.loc[common]
    df_sub_c = df_sub.loc[common]
    models = {}
    for band in bands_OLI:
        X = df_sub_c[[band]].values
        y = df_ref_c[[band]].values
        model = LinearRegression().fit(X, y)
        a, b = model.coef_[0,0], model.intercept_[0]
        models[band] = (a, b)
    internal_models['OLI'][year] = models

# 跨传感器映射模型: TM->OLI 基于参考年样本
df_TM_ref = data_raw[ref_TM].set_index('point_id')
df_OLI_ref = data_raw[ref_OLI].set_index('point_id')
common_pts = df_TM_ref.index.intersection(df_OLI_ref.index)
df_TM_ref_c = df_TM_ref.loc[common_pts]
df_OLI_ref_c = df_OLI_ref.loc[common_pts]

cross_models = {}
# blue, green, red, NIR1
for band in ['blue', 'green', 'red', 'NIR1']:
    X = df_TM_ref_c[[band]].values
    y = df_OLI_ref_c[[band]].values
    model = LinearRegression().fit(X, y)
    cross_models[band] = (model.coef_[0,0], model.intercept_[0])
# MIR -> SWIR1
X_mir = df_TM_ref_c[['MIR']].values
y_swir1 = df_OLI_ref_c[['SWIR1']].values
model_m = LinearRegression().fit(X_mir, y_swir1)
cross_models['MIR2SWIR1'] = (model_m.coef_[0,0], model_m.intercept_[0])

# ---------- 2. 应用模型到影像 ----------
# 处理每个年份影像
for year, img_path in image_paths.items():
    if not os.path.exists(img_path):
        print(f"警告: 影像不存在: {img_path}, 跳过 {year}")
        continue
    with rasterio.open(img_path) as src:
        profile = src.profile.copy()
        # 设定输出为5波段, float32
        profile.update(count=len(common_bands), dtype=rasterio.float32)
        # 读取全部波段到数组 (bands, H, W)
        data = src.read().astype(np.float32)
        # data shape: for TM years: 6 bands; for OLI: 7 bands
        # 内部归一化
        if year in [1998,2003,2008]:
            # TM
            # 若非参考年，做归一化
            if year != ref_TM:
                models = internal_models['TM'][year]
                for i, band in enumerate(bands_TM):
                    a, b = models[band]
                    data[i] = data[i] * a + b
            # 若参考年2003，无需变动
            # 跨传感器映射到 OLI 风格公共波段
            # 创建输出数组
            out_arr = np.full((len(common_bands), data.shape[1], data.shape[2]), np.nan, dtype=np.float32)
            # blue, green, red, NIR1
            for idx, band in enumerate(['blue','green','red','NIR1']):
                band_i = bands_TM.index(band)
                a, b = cross_models[band]
                out_arr[idx] = data[band_i] * a + b
            # SWIR1 from MIR
            mir_i = bands_TM.index('MIR')
            a_m, b_m = cross_models['MIR2SWIR1']
            out_arr[common_bands.index('SWIR1')] = data[mir_i] * a_m + b_m
        else:
            # OLI 年份: [2013,2018]
            # 读取 OLI bands: costal, blue, green, red, NIR1, SWIR1, SWIR2
            # 内部归一化 for 2018
            if year != ref_OLI:
                models = internal_models['OLI'][year]
                for i, band in enumerate(bands_OLI):
                    a, b = models[band]
                    data[i] = data[i] * a + b
            # 参考年 2013: data unchanged
            # 直接选取公共波段
            out_arr = np.full((len(common_bands), data.shape[1], data.shape[2]), np.nan, dtype=np.float32)
            for idx, band in enumerate(common_bands):
                i = bands_OLI.index(band)
                out_arr[idx] = data[i]
        # 输出路径
        out_path = os.path.join(output_dir, f"{year}_calibrated.tif")
        # 写出
        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(out_arr)
    print(f"已处理并保存: {out_path}")

print("所有年份处理完成。")
