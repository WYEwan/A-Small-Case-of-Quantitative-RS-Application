import os
import json
import threading
import sys
import numpy as np
import rasterio
from rasterio.windows import Window
import joblib
import tkinter as tk
from tkinter import ttk, messagebox

# ---------------- 配置路径 ----------------
# 模型与输入影像路径
models_dir = r"models"
# 影像路径映射
image_paths = {
    1998: r"1998.tif",
    2003: r"2003.tif",
    2008: r"2008.tif",
    2013: r"2013.tif",
    2018: r"2018.tif",
}
output_dir = r"D:\通用文件夹"
os.makedirs(output_dir, exist_ok=True)

# 状态文件路径
state_path = os.path.join(output_dir, "progress.json")

# 波段定义
bands_TM = ['blue','green','red','NIR1','NIR2','MIR']
bands_OLI = ['costal_aerosol','blue','green','red','NIR1','SWIR1','SWIR2']
common_bands = ['blue','green','red','NIR1','SWIR1']
# 参考年
ref_TM = 2003
ref_OLI = 2013

# 加载模型
def load_models():
    internal_TM = None
    internal_OLI = None
    cross_TM2OLI = None
    p_tm = os.path.join(models_dir, "internal_TM_RF.pkl")
    p_oli = os.path.join(models_dir, "internal_OLI_RF.pkl")
    p_cross = os.path.join(models_dir, "cross_TM2OLI_RF.pkl")
    if os.path.exists(p_tm):
        internal_TM = joblib.load(p_tm)
    if os.path.exists(p_oli):
        internal_OLI = joblib.load(p_oli)
    if os.path.exists(p_cross):
        cross_TM2OLI = joblib.load(p_cross)
    return internal_TM, internal_OLI, cross_TM2OLI

internal_TM_model, internal_OLI_model, cross_model = load_models()

if internal_TM_model is None and internal_OLI_model is None and cross_model is None:
    raise RuntimeError("未找到任何模型，请先训练并保存模型于指定目录。")

# 加载或初始化进度状态
if os.path.exists(state_path):
    with open(state_path, 'r') as f:
        state = json.load(f)
else:
    # state: { "years": { "1998": {"status":"pending"/"in_progress"/"done", "block": idx}, ... } }
    state = {"years": {}}
    for year in image_paths:
        state["years"][str(year)] = {"status":"pending", "block": 0}
    # 保存初始状态
    with open(state_path, 'w') as f:
        json.dump(state, f)

# GUI 和运行控制
class App:
    def __init__(self, root):
        self.root = root
        root.title("遥感影像校正进度")
        self.pause_flag = False
        self.label = ttk.Label(root, text="准备中...")
        self.label.pack(padx=10, pady=10)
        self.progress = ttk.Progressbar(root, orient="horizontal", length=300, mode="determinate")
        self.progress.pack(padx=10, pady=5)
        self.btn_pause = ttk.Button(root, text="暂停", command=self.on_pause)
        self.btn_pause.pack(padx=10, pady=10)
        # 启动后台线程
        threading.Thread(target=self.run_processing, daemon=True).start()

    def on_pause(self):
        if messagebox.askyesno("暂停确认", "是否暂停并保存当前进度？"):
            self.pause_flag = True

    def run_processing(self):
        try:
            for year_str, info in state["years"].items():
                year = int(year_str)
                if info["status"] == "done":
                    continue
                img_path = image_paths.get(year)
                if img_path is None or not os.path.exists(img_path):
                    state["years"][year_str]["status"] = "done"
                    self.save_state()
                    continue
                # open dataset
                with rasterio.open(img_path) as src:
                    profile = src.profile.copy()
                    height = src.height
                    width = src.width
                    # 输出配置: 波段数 = len(common_bands)
                    profile.update(count=len(common_bands), dtype=rasterio.float32)
                    out_path = os.path.join(output_dir, f"{year}_calibrated_RF.tif")
                    # 如果已存在且状态可能是 done, 跳过读取
                    # 创建目标文件
                    dst = rasterio.open(out_path, 'w', **profile)
                    # 分块大小，可根据内存调整
                    block_size = 512
                    # 计算窗口列表
                    windows = []
                    for i in range(0, height, block_size):
                        h = min(block_size, height - i)
                        for j in range(0, width, block_size):
                            w = min(block_size, width - j)
                            windows.append((i, j, h, w))
                    total = len(windows)
                    start_idx = info.get("block", 0)
                    for idx in range(start_idx, total):
                        if self.pause_flag:
                            # 保存进度并退出
                            state["years"][year_str]["status"] = "in_progress"
                            state["years"][year_str]["block"] = idx
                            self.save_state()
                            dst.close()
                            self.root.quit()
                            return
                        i, j, h, w = windows[idx]
                        window = Window(j, i, w, h)
                        data = src.read(window=window).astype(np.float32)  # shape (bands, h, w)
                        # 内部归一化
                        if year in [1998,2008]:
                            if internal_TM_model is not None:
                                arr = data.reshape(data.shape[0], -1).T  # (n_pixels, bands_TM)
                                arr_norm = internal_TM_model.predict(arr).T.reshape(len(bands_TM), h, w)
                            else:
                                arr_norm = data
                        elif year == ref_TM:
                            arr_norm = data
                        elif year == 2018:
                            if internal_OLI_model is not None:
                                arr = data.reshape(data.shape[0], -1).T
                                arr_norm = internal_OLI_model.predict(arr).T.reshape(len(bands_OLI), h, w)
                            else:
                                arr_norm = data
                        elif year == ref_OLI:
                            arr_norm = data
                        else:
                            arr_norm = data
                        # 跨传感器映射输出 common_bands
                        out_arr = np.full((len(common_bands), h, w), np.nan, dtype=np.float32)
                        if year in [1998,2003,2008] and cross_model is not None:
                            arr = arr_norm.reshape(arr_norm.shape[0], -1).T
                            out_vals = cross_model.predict(arr).T.reshape(len(common_bands), h, w)
                            out_arr = out_vals
                        else:
                            # OLI年份，直接取 common_bands
                            for bi, band in enumerate(common_bands):
                                idx_band = bands_OLI.index(band)
                                out_arr[bi] = arr_norm[idx_band]
                        # 写入窗口
                        dst.write(out_arr, window=window)
                        # 更新进度条
                        percent = int((idx+1)/total*100)
                        self.progress['value'] = percent
                        self.label.config(text=f"年份 {year}: {idx+1}/{total} 块 ({percent}%)")
                        self.root.update_idletasks()
                    # 完成当前年份
                    dst.close()
                    state["years"][year_str]["status"] = "done"
                    state["years"][year_str]["block"] = 0
                    self.save_state()
            # 全部完成
            self.label.config(text="所有年份处理完成")
            self.progress['value'] = 100
            # 短暂停留后关闭窗口
            self.root.after(2000, self.root.destroy)
        except Exception as e:
            messagebox.showerror("错误", str(e))
            self.root.destroy()

    def save_state(self):
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", root.quit)
    root.mainloop()
