import os
import warnings
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window
import matplotlib.pyplot as plt
from scipy.stats import t as student_t
from affine import Affine
from tqdm import tqdm

# 防止中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 输入栅格文件列表，按时间顺序排列
file_paths = [
    r"D:\通用文件夹\定量遥感原理与应用\新数据\新建文件夹\biomass_prediction_with_residuals1998-01.tif",
    r"D:\通用文件夹\定量遥感原理与应用\新数据\新建文件夹\biomass_prediction_with_residuals2003-2003-0.tif",
    r"D:\通用文件夹\定量遥感原理与应用\新数据\新建文件夹\biomass_prediction_with_residuals2008-2008-0.tif",
    r"D:\通用文件夹\定量遥感原理与应用\新数据\新建文件夹\biomass_prediction_with_residuals2013-2013-2.tif",
    r"D:\通用文件夹\定量遥感原理与应用\新数据\新建文件夹\biomass_prediction_with_residuals2018-2018-2 (1).tif",
]
years = np.array([1998, 2003, 2008, 2013, 2018], dtype=np.int32)
ntime = len(file_paths)

# 输出目录
out_dir = r"D:\通用文件夹\定量遥感原理与应用\AGB时间序列分析"
os.makedirs(out_dir, exist_ok=True)

# 检查第一个栅格以获取参考信息
with rasterio.open(file_paths[0]) as ref_src:
    ref_profile = ref_src.profile.copy()
    ref_transform = ref_src.transform
    ref_crs = ref_src.crs
    ref_height = ref_src.height
    ref_width = ref_src.width
    # 获取块窗口生成器
    block_windows = list(ref_src.block_windows())  # 列表 [(idx, window), ...]

# 判断是否具有有效地理参考（CRS 且非 identity transform）
def has_georef(src):
    if src.crs is None:
        return False
    if isinstance(src.transform, Affine) and src.transform == Affine.identity():
        return False
    return True

# 检查所有输入文件地理参考情况
all_georef_ok = True
for fp in file_paths:
    with rasterio.open(fp) as src:
        if not has_georef(src):
            warnings.warn(f"文件 {fp} 缺少地理参考，将采用数组中心裁剪对齐方式", UserWarning)
            all_georef_ok = False
        elif ref_crs is None:
            warnings.warn("参考栅格缺少 CRS 信息，将采用数组中心裁剪对齐方式", UserWarning)
            all_georef_ok = False
        elif src.crs != ref_crs:
            warnings.warn(f"文件 {fp} CRS 与参考不一致，将对其重投影到参考 CRS", UserWarning)
            # 虽然 CRS 不一致，但可重投影；不改变 all_georef_ok
            pass

use_crop = not all_georef_ok

# 若回退裁剪方式，先整体读取并裁剪到最小公共中心尺寸，然后在此裁剪数组上再做块处理
if use_crop:
    # 读取所有为 numpy 数组，仅为裁剪使用；若数据过大，可能需要先手动在 GIS 中裁剪到大致范围
    arrays = []
    shapes = []
    nodata_list = []
    for fp in file_paths:
        with rasterio.open(fp) as src:
            arr = src.read(1).astype(np.float32)
            nd = src.nodata
            if nd is not None:
                arr[arr == nd] = np.nan
            arrays.append(arr)
            shapes.append(arr.shape)
            nodata_list.append(nd)
    # 找到最小高宽
    heights = [s[0] for s in shapes]
    widths = [s[1] for s in shapes]
    min_h = min(heights)
    min_w = min(widths)
    # 中心裁剪所有数组到 (min_h, min_w)
    cropped_arrays = []
    for arr in arrays:
        h, w = arr.shape
        start_row = max(0, (h - min_h)//2)
        start_col = max(0, (w - min_w)//2)
        cropped = arr[start_row:start_row+min_h, start_col:start_col+min_w]
        # 若不足，则填 nan
        ch, cw = cropped.shape
        if ch!=min_h or cw!=min_w:
            tmp = np.full((min_h, min_w), np.nan, dtype=np.float32)
            tmp[:ch, :cw] = cropped
            cropped = tmp
        cropped_arrays.append(cropped)
    # 以裁剪后数组为“参考”，构建简单虚拟 transform（如果原无 georef，此 transform 无意义，仅保留数组索引）
    ref_height, ref_width = min_h, min_w
    # 设一个 identity transform 或保留原第一个的 transform（虽然不精确）
    out_profile = ref_profile.copy()
    out_profile.update(height=ref_height, width=ref_width, dtype=rasterio.float32, count=1, compress='lzw')
    # 生成人工块窗口：按固定块大小（如参考文件的 block size），若无，则自定义，例如 512x512
    # 取参考文件的 block size，如果没有可手动设
    # 这里尝试用 ref_profile 中 blockxsize/blockysize，否则 512
    bsx = ref_profile.get('blockxsize', 512)
    bsy = ref_profile.get('blockysize', 512)
    # 生成窗口列表
    block_windows = []
    for i in range(0, ref_height, bsy):
        hwin = min(bsy, ref_height - i)
        for j in range(0, ref_width, bsx):
            wwin = min(bsx, ref_width - j)
            window = Window(j, i, wwin, hwin)
            block_windows.append((None, window))
    # 为整体平均时序，用累积和与计数
    sum_series = np.zeros(ntime, dtype=np.float64)
    count_series = np.zeros(ntime, dtype=np.int64)
    # 打开输出文件，准备写块
    slope_ds = rasterio.open(os.path.join(out_dir, "AGB_trend_slope.tif"), 'w', **out_profile)
    intercept_ds = rasterio.open(os.path.join(out_dir, "AGB_trend_intercept.tif"), 'w', **out_profile)
    r_ds = rasterio.open(os.path.join(out_dir, "AGB_trend_rvalue.tif"), 'w', **out_profile)
    p_ds = rasterio.open(os.path.join(out_dir, "AGB_trend_pvalue.tif"), 'w', **out_profile)
    pct_ds = rasterio.open(os.path.join(out_dir, f"AGB_pct_change_{years[0]}-{years[-1]}.tif"), 'w', **out_profile)

    # 处理每块
    for _, window in tqdm(block_windows, desc="块处理 (裁剪模式)"):
        win_h = int(window.height); win_w = int(window.width)
        # 从 cropped_arrays 中提取对应窗口的数据，stack 成 (ntime, win_h, win_w)
        block_stack = np.zeros((ntime, win_h, win_w), dtype=np.float32)
        mask_valid_stack = np.zeros((ntime, win_h, win_w), dtype=bool)
        for idx in range(ntime):
            arr = cropped_arrays[idx]
            sub = arr[window.row_off:window.row_off+win_h, window.col_off:window.col_off+win_w]
            block_stack[idx] = sub
            mask_valid_stack[idx] = ~np.isnan(sub)
            # 累积整体平均时序
            valid = mask_valid_stack[idx]
            sum_series[idx] += np.nansum(sub)
            count_series[idx] += np.count_nonzero(valid)
        # 计算斜率等：向量化
        # mask_valid_stack: (ntime, h, w)
        mask = mask_valid_stack
        n = np.sum(mask, axis=0).astype(np.float32)  # (h, w)
        # 只对 n>=2 的像元计算，其他留 nan
        # 计算 sum_x, sum_y, sum_xy, sum_x2, sum_y2
        # years broadcast: (ntime,1,1)
        yrs = years[:,None,None].astype(np.float32)
        sum_x = np.sum(yrs * mask, axis=0)  # (h,w)
        sum_y = np.nansum(block_stack, axis=0)
        sum_xy = np.nansum(block_stack * yrs, axis=0)
        sum_x2 = np.sum((yrs**2) * mask, axis=0)
        sum_y2 = np.nansum(block_stack**2, axis=0)
        # 计算 slope, intercept
        # denom = n*sum_x2 - sum_x**2
        denom = n * sum_x2 - sum_x**2
        slope_blk = np.full((win_h, win_w), np.nan, dtype=np.float32)
        intercept_blk = np.full((win_h, win_w), np.nan, dtype=np.float32)
        r_blk = np.full((win_h, win_w), np.nan, dtype=np.float32)
        p_blk = np.full((win_h, win_w), np.nan, dtype=np.float32)
        valid_trend = denom != 0
        # slope = (n*sum_xy - sum_x*sum_y) / denom
        num = n * sum_xy - sum_x * sum_y
        slope_blk[valid_trend] = (num / denom)[valid_trend]
        # intercept = (sum_y - slope*sum_x) / n
        intercept_blk[valid_trend] = ((sum_y - slope_blk * sum_x) / n)[valid_trend]
        # r: (n*sum_xy - sum_x*sum_y) / sqrt((n*sum_x2 - sum_x^2)*(n*sum_y2 - sum_y^2))
        denom_r = (n * sum_x2 - sum_x**2) * (n * sum_y2 - sum_y**2)
        valid_r = denom_r > 0
        r_blk[valid_r] = ((n * sum_xy - sum_x*sum_y) / np.sqrt(denom_r))[valid_r]
        # p-value: t = r * sqrt((n-2)/(1-r^2)), df = n-2
        # 仅在 valid_r & n>2 & |r|<1 计算
        mask_p = valid_r & (n > 2)
        r_sub = r_blk[mask_p]
        n_sub = n[mask_p]
        # 防止 r_sub 绝对值>=1 导致除零或负值
        r_sub = np.clip(r_sub, -0.999999, 0.999999)
        t_stat = r_sub * np.sqrt((n_sub - 2) / (1 - r_sub**2))
        # 双侧 p-value
        p_blk[mask_p] = 2 * (1 - student_t.cdf(np.abs(t_stat), df=(n_sub - 2)))
        # 计算首尾百分比变化
        first_blk = block_stack[0]
        last_blk = block_stack[-1]
        pct_blk = np.full((win_h, win_w), np.nan, dtype=np.float32)
        valid_pct = (~np.isnan(first_blk)) & (~np.isnan(last_blk)) & (first_blk != 0)
        pct_blk[valid_pct] = (last_blk[valid_pct] - first_blk[valid_pct]) / first_blk[valid_pct] * 100.0
        # 写出各块
        slope_ds.write(slope_blk, 1, window=window)
        intercept_ds.write(intercept_blk, 1, window=window)
        r_ds.write(r_blk, 1, window=window)
        p_ds.write(p_blk, 1, window=window)
        pct_ds.write(pct_blk, 1, window=window)

    # 关闭输出文件
    slope_ds.close()
    intercept_ds.close()
    r_ds.close()
    p_ds.close()
    pct_ds.close()

    # 计算区域平均时序
    mean_series = sum_series / np.where(count_series>0, count_series, 1)
else:
    # 正常地理参考模式：按参考栅格块读取或重投影
    out_profile = ref_profile.copy()
    out_profile.update(dtype=rasterio.float32, count=1, compress='lzw')

    # 打开所有源文件一次，提高效率
    src_list = [rasterio.open(fp) for fp in file_paths]

    # 打开输出文件，准备写块
    slope_ds = rasterio.open(os.path.join(out_dir, "AGB_trend_slope.tif"), 'w', **out_profile)
    intercept_ds = rasterio.open(os.path.join(out_dir, "AGB_trend_intercept.tif"), 'w', **out_profile)
    r_ds = rasterio.open(os.path.join(out_dir, "AGB_trend_rvalue.tif"), 'w', **out_profile)
    p_ds = rasterio.open(os.path.join(out_dir, "AGB_trend_pvalue.tif"), 'w', **out_profile)
    pct_ds = rasterio.open(os.path.join(out_dir, f"AGB_pct_change_{years[0]}-{years[-1]}.tif"), 'w', **out_profile)

    # 为区域平均时序累积
    sum_series = np.zeros(ntime, dtype=np.float64)
    count_series = np.zeros(ntime, dtype=np.int64)

    # 逐块处理
    for idx_win, window in tqdm(block_windows, desc="块处理 (重投影模式)"):
        win_h = int(window.height); win_w = int(window.width)
        # 对每个年份，读取并重投影到参考块
        block_stack = np.zeros((ntime, win_h, win_w), dtype=np.float32)
        mask_valid_stack = np.zeros((ntime, win_h, win_w), dtype=bool)
        # 计算该窗口对应的地理 transform
        dst_transform = rasterio.windows.transform(window, ref_transform)
        for idx, src in enumerate(src_list):
            # 先读取原始数据块或整幅以重投影
            # 直接重投影到目标块数组
            dst_arr = np.full((win_h, win_w), np.nan, dtype=np.float32)
            if not has_georef(src):
                warnings.warn(f"{file_paths[idx]} 缺少地理参考，跳过该波段块", UserWarning)
                # 留 NaN
            else:
                reproject(
                    source=src.read(1),
                    destination=dst_arr,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=ref_crs,
                    resampling=Resampling.bilinear
                )
                # 将 nodata 置为 nan
                nd = src.nodata
                if nd is not None:
                    dst_arr[dst_arr == nd] = np.nan
            block_stack[idx] = dst_arr
            mask_valid_stack[idx] = ~np.isnan(dst_arr)
            # 累积整体平均时序
            valid = mask_valid_stack[idx]
            sum_series[idx] += np.nansum(dst_arr)
            count_series[idx] += np.count_nonzero(valid)

        # 向量化回归计算，与裁剪模式相同
        mask = mask_valid_stack
        n = np.sum(mask, axis=0).astype(np.float32)
        yrs = years[:,None,None].astype(np.float32)
        sum_x = np.sum(yrs * mask, axis=0)
        sum_y = np.nansum(block_stack, axis=0)
        sum_xy = np.nansum(block_stack * yrs, axis=0)
        sum_x2 = np.sum((yrs**2) * mask, axis=0)
        sum_y2 = np.nansum(block_stack**2, axis=0)

        denom = n * sum_x2 - sum_x**2
        slope_blk = np.full((win_h, win_w), np.nan, dtype=np.float32)
        intercept_blk = np.full((win_h, win_w), np.nan, dtype=np.float32)
        r_blk = np.full((win_h, win_w), np.nan, dtype=np.float32)
        p_blk = np.full((win_h, win_w), np.nan, dtype=np.float32)
        valid_trend = denom != 0
        num = n * sum_xy - sum_x * sum_y
        slope_blk[valid_trend] = (num / denom)[valid_trend]
        intercept_blk[valid_trend] = ((sum_y - slope_blk * sum_x) / n)[valid_trend]
        denom_r = (n * sum_x2 - sum_x**2) * (n * sum_y2 - sum_y**2)
        valid_r = denom_r > 0
        r_blk[valid_r] = ((n * sum_xy - sum_x*sum_y) / np.sqrt(denom_r))[valid_r]
        mask_p = valid_r & (n > 2)
        r_sub = r_blk[mask_p]
        n_sub = n[mask_p]
        r_sub = np.clip(r_sub, -0.999999, 0.999999)
        t_stat = r_sub * np.sqrt((n_sub - 2) / (1 - r_sub**2))
        p_blk[mask_p] = 2 * (1 - student_t.cdf(np.abs(t_stat), df=(n_sub - 2)))
        # 百分比变化
        first_blk = block_stack[0]
        last_blk = block_stack[-1]
        pct_blk = np.full((win_h, win_w), np.nan, dtype=np.float32)
        valid_pct = (~np.isnan(first_blk)) & (~np.isnan(last_blk)) & (first_blk != 0)
        pct_blk[valid_pct] = (last_blk[valid_pct] - first_blk[valid_pct]) / first_blk[valid_pct] * 100.0

        # 写块
        slope_ds.write(slope_blk, 1, window=window)
        intercept_ds.write(intercept_blk, 1, window=window)
        r_ds.write(r_blk, 1, window=window)
        p_ds.write(p_blk, 1, window=window)
        pct_ds.write(pct_blk, 1, window=window)

    # 关闭
    for src in src_list:
        src.close()
    slope_ds.close()
    intercept_ds.close()
    r_ds.close()
    p_ds.close()
    pct_ds.close()

    # 区域平均时序
    mean_series = sum_series / np.where(count_series>0, count_series, 1)

# 可视化部分：使用 mean_series 绘制全区平均时序；若需要可另行读取输出栅格作可视化
# 1. 区域平均 AGB 时序
plt.figure(figsize=(6, 4))
plt.plot(years, mean_series, marker='o')
plt.xlabel("年份")
plt.ylabel("平均 AGB")
plt.title("区域平均 AGB 时序变化")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "mean_AGB_timeseries.png"), dpi=300)
plt.close()

# 2. 可视化示例：若想展示趋势斜率全区分布，可读取生成的 TIFF 并显示（此处示例直接用 matplotlib，但可能需要根据 CRS/extent 添加坐标轴、背景图等）
#    这里简单展示如何读取并保存 PNG；也可在 GIS 软件中进一步可视化。
def plot_and_save(tif_path, cmap, title, outname, vmin=None, vmax=None):
    with rasterio.open(tif_path) as ds:
        arr = ds.read(1)
    plt.figure(figsize=(8,6))
    if vmin is not None or vmax is not None:
        plt.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        plt.imshow(arr, cmap=cmap)
    plt.colorbar(label=title)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, outname), dpi=300)
    plt.close()

plot_and_save(os.path.join(out_dir, "AGB_trend_slope.tif"), 'RdYlBu', 'AGB 变化趋势斜率（单位：AGB/年）', "trend_slope.png")
plot_and_save(os.path.join(out_dir, "AGB_trend_pvalue.tif"), 'viridis_r', 'AGB 变化趋势 p-value', "trend_pvalue.png")
plot_and_save(os.path.join(out_dir, "AGB_trend_rvalue.tif"), 'RdBu', 'AGB 变化趋势相关系数 r', "trend_rvalue.png")
plot_and_save(os.path.join(out_dir, f"AGB_pct_change_{years[0]}-{years[-1]}.tif"), 'RdYlGn', f"{years[0]}-{years[-1]} AGB 相对变化百分比(%)", "pct_change.png")

print("时序分析和可视化脚本执行完毕，结果保存在：", out_dir)
