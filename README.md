# A-Small-Case-of-Quantitative-Remote-Sensing-Application

1. Calibration, AGB Modeling, and Time Series Analysis

1.1 Calibration of Multi-Sensor Quantitative Remote Sensing Imagery Based on Linear Regression and Multi-Output Regression with Random Forest

In response to the differences in radiometric response among remote sensing images acquired by different sensors, this study proposes and systematically compares two calibration methods: the classical band-level linear regression calibration and the multi-output regression calibration based on random forest. The former uses common sample points to normalize internally by fitting a linear model to images from different years within the same sensor, and completes cross-sensor calibration between TM and OLI sensors through linear mapping. The latter treats both internal normalization and cross-sensor mapping as a multivariate regression problem, employing random forest ensemble learning to fit all bands simultaneously with a multi-output regressor. Starting from the theoretical framework, this paper provides an in-depth discussion on the model assumptions, algorithm procedures, advantages and disadvantages, and applicable conditions of the two methods, and presents the core formulas for each step with mathematical expressions, providing a solid theoretical foundation for the subsequent experimental results and performance evaluation. Ultimately, this study offers an operational and scalable quantitative calibration approach for long-term sequence analysis of multi-source remote sensing images.

1.2 Quantitative Remote Sensing Estimation of Aboveground Biomass Based on Automated Feature Selection and Multi-Model Comparison

To improve the estimation accuracy of forest aboveground biomass (AGB) from remote sensing data of different years, this paper proposes an automated modeling process integrating multi-step feature selection and multi-model hyperparameter tuning. The process begins with initial feature selection based on correlation coefficients, followed by rapid importance filtering using the median importance threshold of random forest, and finally obtaining the optimal feature subset through RFECV (Recursive Feature Elimination with Cross-Validation). Subsequently, regression models such as Random Forest (RF), XGBoost, LightGBM, and MLP are employed for hyperparameter random search and nested cross-validation, with model stacking evaluation and selection of the optimal model. This method achieved lower RMSE and higher R² on datasets of forest stand samples and spectral values from 1998, 2003, 2008, 2013, and 2018. The experimental results indicate that automated feature selection significantly reduces model complexity, and the multi-model comparison and stacking strategy further enhance predictive performance.

1.3 Temporal and Spatial Trend Analysis of Aboveground Biomass Based on Pixel-Level Linear Regression

This paper addresses the spatiotemporal changes in forest aboveground biomass (AGB) using a time-series trend analysis framework based on pixel-level linear regression. The study first estimates regional AGB using multiple Landsat images and constructs pixel-level time series. It then applies least squares fitting to each pixel to determine the trend of biomass changes over the years, calculating the trend slope, intercept, correlation coefficient (r), significance level (p-value), and percentage change between the start and end periods. Finally, it generates a full remote sensing raster data output and extracts regional average time-series curves for visualization analysis. Taking a typical forest area's AGB data from five periods between 1998 and 2018 as an example, the results show a significant upward/downward trend in overall biomass with spatial heterogeneity across different ecological zones. This method, characterized by its simple parameters and large-scale automation capability, provides a quantitative basis for forest carbon dynamics monitoring and management decision-making.

定标、AGB建模、时序分析

1.基于线性回归与随机森林多输出回归的多传感器定量遥感影像定标方法
针对不同传感器获取的遥感影像在辐射响应上的差异，本研究提出并系统比较了两类定标方法：一是经典的波段级线性回归定标，二是基于随机森林的多输出回归定标。前者利用公共样本点，通过对同传感器内不同年份影像进行线性模型拟合，实现内部归一化；并通过线性映射完成TM与OLI两传感器间跨传感器定标。后者则将内部归一化与跨传感器映射均视为多变量回归问题，采用随机森林集成学习，通过多输出回归器同时对所有波段进行拟合。本文从理论框架出发，对两种方法的模型假设、算法流程、优缺点及适用条件进行了深入阐述，并结合数学表达式给出了各步骤的核心公式，为后续实验结果与性能评估提供了坚实的理论支撑。最终，本研究为多源遥感影像长期序列分析提供了一种可操作、可扩展的定量定标思路。

2.基于自动化特征选择与多模型比较的定量遥感地上生物量估计
为了提高不同年份遥感数据下森林地上生物量（AGB）的估计精度，本文提出了一种集成多步特征选择与多模型超参数调优的自动化建模流程。该流程首先基于相关系数筛选获得初步特征集，继而利用随机森林中位数重要性阈值进行快速重要性过滤，再通过 RFECV（递归特征消除与交叉验证）得到最终最优特征子集；随后采用随机森林（RF）、XGBoost、LightGBM、MLP 等回归模型进行超参数随机搜索与嵌套交叉验证，并结合模型堆叠（Stacking）评估与选择最优模型。该方法在 1998、2003、2008、2013、2018 年林分样本及光谱值数据集上均取得了较低的 RMSE 和更高的 R²。实验结果表明，自动化特征选择显著降低了模型复杂度，且多模型比较及堆叠策略能够进一步提升预测性能。

3.基于像元线性回归的多时相以上生物量时空变化趋势分析
本文针对森林地上生物量（Aboveground Biomass, AGB）时空变化，使用了一种基于像元级线性回归的时序趋势分析框架。研究首先利用多期Landsat影像估计区域AGB，并构建像元级时间序列；随后对每个像元应用最小二乘法拟合生物量随年份的变化趋势，计算趋势斜率、截距、相关系数（r）、显著性水平（p值）及首尾百分比变化；最后生成整幅遥感栅格数据输出，并提取区域平均时序曲线进行可视化分析。以某典型森林区1998–2018年五个时期AGB数据为例，结果显示整体生物量呈显著上升/下降趋势，并在不同生态分区表现出空间异质性。该方法具有参数简单、可大规模自动化处理的优点，为森林碳动态监测与管理决策提供量化依据。

References:
参考文献：

1.Calibration module
1.定标模块
[1] Chander, G., Markham, B. L., & Helder, D. L. (2009). Summary of current radiometric calibration coefficients for Landsat MSS, TM, ETM+, and EO-1 ALI sensors. Remote sensing of environment, 113(5), 893-903.
[2] Slater, P. N., Biggar, S. F., Palmer, J. M., & Thome, K. J. (2001). Unified approach to absolute radiometric calibration in the solar-reflective range. Remote Sensing of Environment, 77(3), 293-303.
[3] Breiman, L. (2001). Random forests. Machine learning, 45, 5-32.
[4] Belgiu, M., & Drăguţ, L. (2016). Random forest in remote sensing: A review of applications and future directions. ISPRS journal of photogrammetry and remote sensing, 114, 24-31.

2.AGB modeling module
2.AGB建模
[1] Lu, D., Chen, Q., Wang, G., Liu, L., Li, G., & Moran, E. (2016). A survey of remote sensing-based aboveground biomass estimation methods in forest ecosystems. International Journal of Digital Earth, 9(1), 63-105.
[2] Tian, L., Wu, X., Tao, Y., Li, M., Qian, C., Liao, L., & Fu, W. (2023). Review of remote sensing-based methods for forest aboveground biomass estimation: Progress, challenges, and prospects. Forests, 14(6), 1086.
[3] Zhou, J., Zan, M., Zhai, L., Yang, S., Xue, C., Li, R., & Wang, X. (2025). Remote sensing estimation of aboveground biomass of different forest types in Xinjiang based on machine learning. Scientific Reports, 15(1), 6187.
[4] Breiman, L. (2001). Random forests. Machine learning, 45, 5-32.
[5] Chen, T., & Guestrin, C. (2016, August). Xgboost: A scalable tree boosting system. In Proceedings of the 22nd acm sigkdd international conference on knowledge discovery and data mining (pp. 785-794).
[6] Guyon, I., & Elisseeff, A. (2003). An introduction to variable and feature selection. Journal of machine learning research, 3(Mar), 1157-1182. 

3.Timing analysis module
3.时序
[1] Saatchi, S. S., Harris, N. L., Brown, S., Lefsky, M., Mitchard, E. T., Salas, W., ... & Morel, A. (2011). Benchmark map of forest carbon stocks in tropical regions across three continents. Proceedings of the national academy of sciences, 108(24), 9899-9904.
[2] Asner, G. P., Powell, G. V., Mascaro, J., Knapp, D. E., Clark, J. K., Jacobson, J., ... & Hughes, R. F. (2010). High-resolution forest carbon stocks and emissions in the Amazon. Proceedings of the National Academy of Sciences, 107(38), 16738-16742.
[3] Verbesselt, J., Hyndman, R., Newnham, G., & Culvenor, D. (2010). Detecting trend and seasonal changes in satellite image time series. Remote sensing of Environment, 114(1), 106-115.
[4] Zhu, Z., & Woodcock, C. E. (2014). Continuous change detection and classification of land cover using all available Landsat data. Remote sensing of Environment, 144, 152-171.
[5] Kennedy, R. E., Yang, Z., & Cohen, W. B. (2010). Detecting trends in forest disturbance and recovery using yearly Landsat time series: 1. LandTrendr—Temporal segmentation algorithms. Remote Sensing of Environment, 114(12), 2897-2910.

