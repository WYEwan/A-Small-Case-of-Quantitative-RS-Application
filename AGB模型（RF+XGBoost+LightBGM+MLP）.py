import os
import re
import pandas as pd
import numpy as np
# 在导入 pyplot 之前设置后端为 Agg，避免 tkinter 等 GUI 后端报错
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, cross_validate, learning_curve
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.feature_selection import RFECV, SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
import joblib
import datetime
import warnings

warnings.filterwarnings("ignore")

# 设置中文字体，防止中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 尝试导入 XGBoost 与 LightGBM
have_xgb = False
have_lgb = False
try:
    from xgboost import XGBRegressor
    have_xgb = True
except ImportError:
    print("未安装 xgboost，将跳过 XGBoost 模型。")
try:
    from lightgbm import LGBMRegressor
    have_lgb = True
except ImportError:
    print("未安装 lightgbm，将跳过 LightGBM 模型。")

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def save_plot(fig, output_dir, name):
    """保存 matplotlib Figure 到 output_dir，文件名 name（不含扩展名），使用 PNG 格式。"""
    filepath = os.path.join(output_dir, f"{name}.png")
    fig.savefig(filepath, bbox_inches='tight')
    plt.close(fig)

def extract_year_from_filename(path):
    """从文件名中提取4位年份，若找不到则返回文件名作为标识。"""
    name = os.path.splitext(os.path.basename(path))[0]
    m = re.search(r"(19|20)\d{2}", name)
    if m:
        return m.group(0)
    else:
        return name

def process_single_dataset(data_path, base_output_dir):
    year = extract_year_from_filename(data_path)
    output_dir = os.path.join(base_output_dir, year)
    ensure_dir(output_dir)

    # 1. 读取数据
    df = pd.read_csv(data_path)
    if 'ID' in df.columns:
        id_series = df['ID']
    else:
        id_series = pd.Series(np.arange(len(df)), name='ID')
    if 'AGB' not in df.columns:
        raise ValueError(f"{data_path} 中未找到列 'AGB'")
    y = df['AGB'].values
    X = df.drop(columns=[c for c in ['ID', 'AGB'] if c in df.columns])
    feature_names = list(X.columns)

    # 2. 缺失值处理
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=feature_names)

    # 3. 训练/测试划分
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X_imputed, y, id_series, test_size=0.2, random_state=42
    )
    # 保存划分信息
    df_train_split = pd.DataFrame({'ID': id_train.values, '集': ['训练集'] * len(id_train)})
    df_test_split = pd.DataFrame({'ID': id_test.values, '集': ['测试集'] * len(id_test)})
    split_info = pd.concat([df_train_split, df_test_split], ignore_index=True)
    split_info.to_csv(os.path.join(output_dir, "train_test_split_info.csv"), index=False, encoding='utf-8')

    # 4. 特征相关度初筛
    df_train_full = pd.DataFrame(X_train.copy(), columns=feature_names)
    df_train_full['AGB'] = y_train
    corr_with_target = df_train_full.corr()['AGB'].drop('AGB')
    corr_abs = corr_with_target.abs().sort_values(ascending=False)
    # 可视化 Top 30 相关特征条形
    top_n = min(30, len(corr_abs))
    top_feats = corr_abs.index[:top_n]
    fig = plt.figure(figsize=(10,6))
    colors = plt.cm.Pastel1.colors
    plt.bar(range(top_n), corr_abs[top_feats], color=colors[:top_n])
    plt.xticks(range(top_n), top_feats, rotation=90)
    plt.ylabel("与 AGB 的绝对相关系数")
    plt.title(f"{year}：前 {top_n} 特征相关度排序")
    save_plot(fig, output_dir, "相关度条形图Top特征")

    # 相关矩阵热图（特征过多时可注释此块）
    corr_matrix = df_train_full.corr()
    fig = plt.figure(figsize=(12,10))
    im = plt.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    ticks = range(len(corr_matrix.columns))
    plt.xticks(ticks, corr_matrix.columns, rotation=90)
    plt.yticks(ticks, corr_matrix.columns)
    plt.title(f"{year}：特征与 AGB 相关矩阵热图")
    save_plot(fig, output_dir, "特征相关矩阵热图")

    # 初步筛选阈值
    corr_threshold = 0.1
    prelim_selected = corr_abs[corr_abs >= corr_threshold].index.tolist()
    if len(prelim_selected) < 5:
        prelim_selected = corr_abs.index[:min(10, len(corr_abs))].tolist()
    pd.DataFrame({'feature': prelim_selected}).to_csv(
        os.path.join(output_dir, "初步相关度筛选特征列表.csv"), index=False, encoding='utf-8'
    )

    # 5. 快速重要性过滤
    X_train_prelim = X_train[prelim_selected]
    rf_quick = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_quick.fit(X_train_prelim, y_train)
    selector = SelectFromModel(rf_quick, threshold='median', prefit=True)
    features_after_imp = [f for f, flag in zip(prelim_selected, selector.get_support()) if flag]
    if len(features_after_imp) < 3:
        features_after_imp = corr_abs.index[:min(5, len(corr_abs))].tolist()
    pd.DataFrame({'feature': features_after_imp}).to_csv(
        os.path.join(output_dir, "重要性过滤后特征列表.csv"), index=False, encoding='utf-8'
    )
    # 可视化特征数量对比
    fig = plt.figure(figsize=(6,4))
    plt.bar(['初步筛选','重要性过滤后'], [len(prelim_selected), len(features_after_imp)],
            color=['lightblue','lightgreen'])
    plt.ylabel("特征数量")
    plt.title(f"{year}：初筛 vs 重要性过滤后 特征数量")
    save_plot(fig, output_dir, "特征数量对比")

    # 6. RFECV 递归精筛
    rf_estimator = RandomForestRegressor(random_state=42, n_jobs=-1)
    rfecv = RFECV(
        estimator=rf_estimator,
        step=1,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    print(f"{year}：开始 RFECV 特征选择……")
    rfecv.fit(X_train[features_after_imp], y_train)
    selected_mask = rfecv.support_
    selected_features = [f for f, flag in zip(features_after_imp, selected_mask) if flag]
    pd.DataFrame({'feature': selected_features}).to_csv(
        os.path.join(output_dir, "最终选中特征列表.csv"), index=False, encoding='utf-8'
    )
    # RFECV 过程可视化
    scores = None
    if hasattr(rfecv, "grid_scores_"):
        try:
            scores = rfecv.grid_scores_
        except:
            scores = None
    if scores is None and hasattr(rfecv, "cv_results_"):
        try:
            scores = rfecv.cv_results_['mean_test_score']
        except:
            scores = None
    if scores is not None:
        fig = plt.figure(figsize=(8,5))
        num_feats = np.arange(1, len(scores)+1)
        plt.plot(num_feats, scores, marker='o', linestyle='-', color='teal')
        plt.xlabel("保留特征个数")
        plt.ylabel("CV 得分 (neg MSE)")
        plt.title(f"{year}：RFECV 特征数 vs CV 得分")
        save_plot(fig, output_dir, "RFECV特征数_vs_CV得分")

    # 7. 模型候选与超参调优
    X_train_sel = X_train[selected_features]
    X_test_sel = X_test[selected_features]
    cv_outer = KFold(n_splits=5, shuffle=True, random_state=42)

    model_results = {}

    def tune_and_cv(estimator, param_dist, name):
        rnd = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_dist,
            n_iter=20,
            scoring='neg_mean_squared_error',
            cv=cv_outer,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        rnd.fit(X_train_sel, y_train)
        best = rnd.best_estimator_
        cv_res = cross_validate(best, X_train_sel, y_train, cv=cv_outer,
                                scoring={'RMSE':'neg_mean_squared_error','MAE':'neg_mean_absolute_error','R2':'r2'},
                                return_train_score=False)
        rmse_scores = np.sqrt(-cv_res['test_RMSE'])
        mae_scores = -cv_res['test_MAE']
        r2_scores = cv_res['test_R2']
        model_results[name] = {
            'estimator': best,
            'rmse_scores': rmse_scores,
            'mae_scores': mae_scores,
            'r2_scores': r2_scores
        }
        return best, rmse_scores

    # RandomForest
    param_rf = {
        'n_estimators': [100,200,300],
        'max_depth': [None,10,20,30],
        'min_samples_split': [2,5,10],
        'min_samples_leaf': [1,2,4],
        'max_features': ['sqrt','log2']
    }
    print(f"{year}：调优 RandomForest……")
    best_rf, _ = tune_and_cv(RandomForestRegressor(random_state=42), param_rf, 'RF')

    # XGBoost
    if have_xgb:
        param_xgb = {
            'n_estimators': [100,200,300],
            'max_depth': [3,5,7,10],
            'learning_rate': [0.01,0.05,0.1],
            'subsample': [0.7,0.8,1.0],
            'colsample_bytree': [0.7,0.8,1.0]
        }
        print(f"{year}：调优 XGBoost……")
        best_xgb, _ = tune_and_cv(XGBRegressor(random_state=42, verbosity=0), param_xgb, 'XGB')

    # LightGBM
    if have_lgb:
        param_lgb = {
            'n_estimators': [100,200,300],
            'num_leaves': [31,50,100],
            'learning_rate': [0.01,0.05,0.1],
            'subsample': [0.7,0.8,1.0],
            'colsample_bytree': [0.7,0.8,1.0]
        }
        print(f"{year}：调优 LightGBM……")
        best_lgb, _ = tune_and_cv(LGBMRegressor(random_state=42), param_lgb, 'LGB')

    # MLPRegressor，early_stopping=True
    param_mlp = {
        'hidden_layer_sizes': [(50,), (100,), (100,50)],
        'activation': ['relu','tanh'],
        'alpha': [0.0001, 0.001],
        'learning_rate_init': [0.001, 0.01]
    }
    print(f"{year}：调优 MLPRegressor（early_stopping=True）……")
    mlp_base = MLPRegressor(max_iter=1000, early_stopping=True, validation_fraction=0.1, random_state=42)
    best_mlp, _ = tune_and_cv(mlp_base, param_mlp, 'MLP')

    # 可视化各模型 CV RMSE 分布箱线
    names = list(model_results.keys())
    data_rmse = [model_results[n]['rmse_scores'] for n in names]
    fig = plt.figure(figsize=(8,6))
    bp = plt.boxplot(data_rmse, labels=names, patch_artist=True)
    colors = plt.cm.Pastel2.colors
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    plt.ylabel("RMSE")
    plt.title(f"{year}：各模型 CV RMSE 分布对比")
    save_plot(fig, output_dir, "模型CV_RMSE对比")

    # 选择最佳模型
    mean_rmses = {name: model_results[name]['rmse_scores'].mean() for name in names}
    best_name = min(mean_rmses, key=mean_rmses.get)
    best_model = model_results[best_name]['estimator']
    print(f"{year}：最佳模型为 {best_name}，平均 CV RMSE = {mean_rmses[best_name]:.4f}")
    joblib.dump(best_model, os.path.join(output_dir, f"最佳模型_{best_name}.joblib"))

    # 可选堆叠
    if len(names) >= 2:
        sorted_by_rmse = sorted(mean_rmses.items(), key=lambda x: x[1])
        top2 = sorted_by_rmse[:2]
        estimators = [(nm, model_results[nm]['estimator']) for nm,_ in top2]
        stack = StackingRegressor(estimators=estimators, final_estimator=Ridge(), cv=cv_outer, n_jobs=-1)
        print(f"{year}：评估堆叠模型……")
        stack_cv = cross_validate(stack, X_train_sel, y_train, cv=cv_outer,
                                  scoring={'RMSE':'neg_mean_squared_error','MAE':'neg_mean_absolute_error','R2':'r2'},
                                  return_train_score=False)
        stack_rmse = np.sqrt(-stack_cv['test_RMSE'])
        mean_stack_rmse = stack_rmse.mean()
        fig = plt.figure(figsize=(6,4))
        bpp = plt.boxplot([stack_rmse], labels=['Stack'], patch_artist=True)
        for patch in bpp['boxes']:
            patch.set_facecolor('lightcyan')
        plt.title(f"{year}：堆叠模型 CV RMSE 分布 (均值 {mean_stack_rmse:.4f})")
        save_plot(fig, output_dir, "堆叠模型CV_RMSE")
        if mean_stack_rmse < mean_rmses[best_name]:
            print(f"{year}：堆叠模型优于单模型，替换最佳模型")
            best_model = stack
            best_name = 'Stack'
            stack.fit(X_train_sel, y_train)
            joblib.dump(best_model, os.path.join(output_dir, f"最佳模型_{best_name}.joblib"))

    # 8. 测试集评估，收集所有候选模型表现
    test_metrics = {}
    for name, res in model_results.items():
        est = res['estimator']
        y_pred = est.predict(X_test_sel)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        test_metrics[name] = {'RMSE': rmse, 'MAE': mae, 'R2': r2}
    if 'stack' in locals() and best_name == 'Stack':
        y_pred = best_model.predict(X_test_sel)
        test_metrics['Stack'] = {
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE': mean_absolute_error(y_test, y_pred),
            'R2': r2_score(y_test, y_pred)
        }
    # 保存各模型测试集比较
    df_test_all = pd.DataFrame(test_metrics).T.reset_index().rename(columns={'index':'模型'})
    df_test_all.to_csv(os.path.join(output_dir, "各模型测试集表现比较.csv"), index=False, encoding='utf-8')

    # 9. 绘制“大图”：比较各模型测试集 RMSE/MAE/R²
    models = list(df_test_all['模型'])
    x = np.arange(len(models))
    width = 0.25
    fig = plt.figure(figsize=(10,6))
    plt.bar(x - width, df_test_all['RMSE'], width, label='RMSE', color=plt.cm.Pastel2.colors[0])
    plt.bar(x, df_test_all['MAE'], width, label='MAE', color=plt.cm.Pastel2.colors[1])
    plt.bar(x + width, df_test_all['R2'], width, label='R²', color=plt.cm.Pastel2.colors[2])
    plt.xticks(x, models, rotation=45)
    plt.xlabel("模型")
    plt.title(f"{year}：各模型测试集表现比较")
    plt.legend()
    plt.tight_layout()
    save_plot(fig, output_dir, "各模型测试集表现大图")

    # 10. 针对最佳模型绘制详细图表
    y_test_pred = best_model.predict(X_test_sel)
    fig = plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_test_pred, alpha=0.6, color='orange')
    mn, mx = min(np.min(y_test), np.min(y_test_pred)), max(np.max(y_test), np.max(y_test_pred))
    plt.plot([mn,mx],[mn,mx],'r--')
    plt.xlabel("真实 AGB"); plt.ylabel("预测 AGB")
    plt.title(f"{year}：测试集 真实 vs 预测 ({best_name})")
    save_plot(fig, output_dir, "测试集真实_vs_预测散点")

    fig = plt.figure(figsize=(8,5))
    residuals_test = y_test - y_test_pred
    plt.hist(residuals_test, bins=30, edgecolor='black', color='lightblue')
    plt.xlabel("残差 (真实 - 预测)"); plt.ylabel("频数")
    plt.title(f"{year}：测试集残差分布 ({best_name})")
    save_plot(fig, output_dir, "测试集残差直方图")

    try:
        from sklearn.inspection import permutation_importance
        perm_imp = permutation_importance(best_model, X_test_sel, y_test, n_repeats=10, random_state=42, n_jobs=-1)
        imp_means = perm_imp.importances_mean
        sorted_idx = np.argsort(imp_means)[::-1]
        feat_sorted = [selected_features[i] for i in sorted_idx]
        imp_sorted = imp_means[sorted_idx]
        pd.DataFrame({'feature': feat_sorted, 'perm_importance': imp_sorted}).to_csv(
            os.path.join(output_dir, "PermutationImportance.csv"), index=False, encoding='utf-8'
        )
        topk = min(20, len(feat_sorted))
        fig = plt.figure(figsize=(10,6))
        plt.bar(range(topk), imp_sorted[:topk], color=plt.cm.Pastel2.colors[:topk])
        plt.xticks(range(topk), feat_sorted[:topk], rotation=90)
        plt.ylabel("Permutation Importance 平均值")
        plt.title(f"{year}：Permutation Importance 前 {topk} 特征 ({best_name})")
        save_plot(fig, output_dir, "PermutationImportanceTop特征")
    except Exception:
        pass

    try:
        train_sizes, train_scores, valid_scores = learning_curve(
            best_model, X_train_sel, y_train,
            cv=cv_outer,
            scoring='neg_mean_squared_error',
            train_sizes=np.linspace(0.1,1.0,5),
            n_jobs=-1
        )
        train_rmse = np.sqrt(-train_scores)
        valid_rmse = np.sqrt(-valid_scores)
        train_mean = train_rmse.mean(axis=1); train_std = train_rmse.std(axis=1)
        val_mean = valid_rmse.mean(axis=1); val_std = valid_rmse.std(axis=1)
        fig = plt.figure(figsize=(8,6))
        plt.plot(train_sizes, train_mean, label='训练 RMSE', color='orange')
        plt.fill_between(train_sizes, train_mean-train_std, train_mean+train_std, color='orange', alpha=0.3)
        plt.plot(train_sizes, val_mean, label='验证 RMSE', color='purple')
        plt.fill_between(train_sizes, val_mean-val_std, val_mean+val_std, color='purple', alpha=0.3)
        plt.xlabel("训练样本比例"); plt.ylabel("RMSE")
        plt.title(f"{year}：学习曲线 ({best_name})")
        plt.legend()
        save_plot(fig, output_dir, "学习曲线_RMSE")
    except Exception:
        pass

    # 11. 保存最佳模型测试集指标 & CV 指标 & 日志
    with open(os.path.join(output_dir, "最佳模型测试集评估指标.txt"), 'w', encoding='utf-8') as f:
        tm = test_metrics.get(best_name, {'RMSE':np.nan,'MAE':np.nan,'R2':np.nan})
        f.write(f"{year} 最佳模型: {best_name}\n")
        f.write(f"测试集 RMSE: {tm['RMSE']:.4f}\n")
        f.write(f"测试集 MAE: {tm['MAE']:.4f}\n")
        f.write(f"测试集 R²: {tm['R2']:.4f}\n")
    # 保存 CV 各折指标
    if best_name in model_results:
        res = model_results[best_name]
        df_cv = pd.DataFrame({
            '折次': np.arange(1, len(res['rmse_scores'])+1),
            'RMSE': res['rmse_scores'],
            'MAE': res['mae_scores'],
            'R2': res['r2_scores']
        })
    else:
        stack_cv = cross_validate(best_model, X_train_sel, y_train, cv=cv_outer,
                                  scoring={'RMSE':'neg_mean_squared_error','MAE':'neg_mean_absolute_error','R2':'r2'},
                                  return_train_score=False)
        df_cv = pd.DataFrame({
            '折次': np.arange(1, len(stack_cv['test_R2'])+1),
            'RMSE': np.sqrt(-stack_cv['test_RMSE']),
            'MAE': -stack_cv['test_MAE'],
            'R2': stack_cv['test_R2']
        })
    df_cv.to_csv(os.path.join(output_dir, "最佳模型交叉验证各折指标.csv"), index=False, encoding='utf-8')

    # 运行日志
    log_path = os.path.join(output_dir, "运行日志.txt")
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"运行时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"数据路径: {data_path}\n")
        f.write(f"初筛特征数: {len(prelim_selected)}, 重要性过滤后: {len(features_after_imp)}, RFECV后: {len(selected_features)}\n")
        for name in model_results:
            f.write(f"{name} 平均 CV RMSE: {model_results[name]['rmse_scores'].mean():.4f}\n")
        f.write(f"最终最佳模型: {best_name}\n")
        tm = test_metrics.get(best_name, {'RMSE':np.nan,'MAE':np.nan,'R2':np.nan})
        f.write(f"测试集 RMSE: {tm['RMSE']:.4f}, MAE: {tm['MAE']:.4f}, R²: {tm['R2']:.4f}\n")
        f.write("-"*50 + "\n")

    return {
        'year': year,
        'best_model': best_name,
        'num_features_selected': len(selected_features),
        'test_rmse': test_metrics.get(best_name, {'RMSE':np.nan})['RMSE'],
        'test_mae': test_metrics.get(best_name, {'MAE':np.nan})['MAE'],
        'test_r2': test_metrics.get(best_name, {'R2':np.nan})['R2']
    }

def main():
    data_paths = [
        r"D:\通用文件夹\定量遥感原理与应用\样本点及其光谱值\1998withID.csv",
        r"D:\通用文件夹\定量遥感原理与应用\样本点及其光谱值\2003withID.csv",
        r"D:\通用文件夹\定量遥感原理与应用\样本点及其光谱值\2008withID.csv",
        r"D:\通用文件夹\定量遥感原理与应用\样本点及其光谱值\2013withID.csv",
        r"D:\通用文件夹\定量遥感原理与应用\样本点及其光谱值\2018withID.csv"
    ]
    base_output_dir = r"D:\通用文件夹\定量遥感原理与应用\AGB模型\高级模型对比"
    ensure_dir(base_output_dir)

    summary_list = []
    for path in data_paths:
        print(f"处理: {path}")
        try:
            res = process_single_dataset(path, base_output_dir)
            summary_list.append(res)
        except Exception as e:
            print(f"{path} 处理出错: {e}")

    # 汇总对比
    if summary_list:
        df_summary = pd.DataFrame(summary_list).sort_values('year')
        df_summary.to_csv(os.path.join(base_output_dir, "各年份最佳模型汇总指标.csv"), index=False, encoding='utf-8')
        years = df_summary['year'].astype(str).tolist()
        x = np.arange(len(years))

        # 可视化测试集 RMSE/MAE/R2 对比
        fig = plt.figure(figsize=(8,5))
        plt.bar(x, df_summary['test_rmse'], color=plt.cm.Pastel2.colors[:len(x)])
        plt.xticks(x, years)
        plt.xlabel("年份"); plt.ylabel("测试集 RMSE")
        plt.title("各年份测试集 RMSE 对比")
        save_plot(fig, base_output_dir, "各年份测试集RMSE对比")

        fig = plt.figure(figsize=(8,5))
        plt.bar(x, df_summary['test_mae'], color=plt.cm.Pastel2.colors[:len(x)])
        plt.xticks(x, years)
        plt.xlabel("年份"); plt.ylabel("测试集 MAE")
        plt.title("各年份测试集 MAE 对比")
        save_plot(fig, base_output_dir, "各年份测试集MAE对比")

        fig = plt.figure(figsize=(8,5))
        plt.bar(x, df_summary['test_r2'], color=plt.cm.Pastel2.colors[:len(x)])
        plt.xticks(x, years)
        plt.xlabel("年份"); plt.ylabel("测试集 R²")
        plt.title("各年份测试集 R² 对比")
        save_plot(fig, base_output_dir, "各年份测试集R2对比")

        # 最佳模型类型分布
        fig = plt.figure(figsize=(6,4))
        counts = df_summary['best_model'].value_counts()
        plt.bar(counts.index, counts.values, color=plt.cm.Pastel2.colors[:len(counts)])
        plt.ylabel("数量"); plt.title("各年份最佳模型类型分布")
        save_plot(fig, base_output_dir, "最佳模型类型分布")

    print("所有年份处理完成，结果保存在:", base_output_dir)

if __name__ == '__main__':
    main()
