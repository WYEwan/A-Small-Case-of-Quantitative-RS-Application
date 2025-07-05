import geopandas as gpd
import rasterio
import pandas as pd
import numpy as np
from shapely.geometry import Point
import re

# -----------------------------
# 1. 定义输入路径
# -----------------------------
shp_path = r"不变点.shp"
tif_paths = {
    "1998": r"1998.tif",
    "2003": r"2003.tif",
    "2008": r"2008.tif",
    "2013": r"2013.tif",
    "2018": r"2018.tif",
}

# -----------------------------
# 2. 读取 SHP 点数据
# -----------------------------
gdf = gpd.read_file(shp_path)
gdf = gdf.to_crs("EPSG:4326")  # 可根据栅格坐标系调整
gdf['point_id'] = gdf.index  # 添加索引字段

# -----------------------------
# 3. 遍历每个 TIF，提取点的像元值
# -----------------------------
all_data = pd.DataFrame({'point_id': gdf['point_id']})

for year, tif_path in tif_paths.items():
    with rasterio.open(tif_path) as src:
        coords = [(geom.x, geom.y) for geom in gdf.geometry]
        band_count = src.count
        values = []
        for val in src.sample(coords):
            values.append(val)
        values = np.array(values)  # shape: (n_points, n_bands)
        
        # 获取栅格文件的波段描述
        band_descriptions = src.descriptions
        
        # 构造 DataFrame 列名
        for b in range(band_count):
            # 使用原始波段描述
            band_name = band_descriptions[b] if b < len(band_descriptions) and band_descriptions[b] else f"Band_{b+1}"
            
            # 清理名称：替换特殊字符，确保列名有效性
            band_name = re.sub(r'[^a-zA-Z0-9_]+', '_', band_name)
            
            # 如果清理后为空，使用默认名称
            if not band_name.strip():
                band_name = f"Band_{b+1}"
            
            all_data[f"{year}_{band_name}"] = values[:, b]

# -----------------------------
# 4. 导出为 CSV 表格
# -----------------------------
all_data.to_csv("point_raster_values.csv", index=False)
print("已完成提取，输出文件为 point_raster_values.csv")