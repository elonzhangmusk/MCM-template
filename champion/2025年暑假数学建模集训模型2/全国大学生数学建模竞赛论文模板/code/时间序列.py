import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.tsa.seasonal import seasonal_decompose
import itertools
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 1. 数据加载与预处理
# 读取数据（请确保文件路径正确）
try:
    # 尝试读取Excel文件
    ChinaBank = pd.read_excel('sjxl2.xlsx', index_col='Date', parse_dates=['Date'])
except:
    # 如果Excel文件不存在，尝试读取CSV文件
    ChinaBank = pd.read_csv('ChinaBank.csv', index_col='Date', parse_dates=['Date'])

# 确保索引是 datetime 类型
ChinaBank.index = pd.to_datetime(ChinaBank.index)

# 查看数据基本信息
print("数据基本信息：")
print(ChinaBank.info())
print("\n数据前5行：")
print(ChinaBank.head())

# 选择时间范围和收盘价列
sub = ChinaBank.loc['2025-01-07':'2025-04-05', 'Close'].dropna()
print("\n筛选后的收盘价数据形状：", sub.shape)

# 划分训练集和测试集
train = sub.loc['2025-01':'2025-02']
test = sub.loc['2025-03':'2025-04']

print(f"\n训练集大小: {len(train)}, 测试集大小: {len(test)}")

# 2. 数据可视化
# 绘制训练集数据
plt.figure(figsize=(12, 6))
plt.plot(train, label='训练集')
plt.title('训练集收盘价时间序列')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# 3. 数据平稳性处理与检验
# 计算差分
ChinaBank['diff_1'] = ChinaBank['Close'].diff(1)  # 1阶差分
ChinaBank['diff_2'] = ChinaBank['diff_1'].diff(1)  # 2阶差分

# 填充缺失值
ChinaBank['diff_1'] = ChinaBank['diff_1'].fillna(0)
ChinaBank['diff_2'] = ChinaBank['diff_2'].fillna(0)

# 绘制原序列与差分序列
fig = plt.figure(figsize=(12, 10))
ax1 = fig.add_subplot(311)
ax1.plot(ChinaBank['Close'], label='原始序列')
ax1.set_title('原始序列')
ax1.legend()

ax2 = fig.add_subplot(312)
ax2.plot(ChinaBank['diff_1'], label='1阶差分', color='orange')
ax2.set_title('1阶差分序列')
ax2.legend()

ax3 = fig.add_subplot(313)
ax3.plot(ChinaBank['diff_2'], label='2阶差分', color='green')
ax3.set_title('2阶差分序列')
ax3.legend()

plt.tight_layout()
plt.show()


# 单位根检验(ADF检验)
def adf_test(series, title=''):
    print(f'=== {title} 的ADF检验结果 ===')
    result = ADF(series)
    labels = ['ADF统计量', 'p值', '滞后阶数', '观测值数量']
    for label, value in zip(labels, result):
        print(f'{label}: {value:.4f}')
    if result[1] <= 0.05:
        print("结论: 拒绝原假设，序列是平稳的")
    else:
        print("结论: 不能拒绝原假设，序列是非平稳的")


adf_test(ChinaBank['Close'].dropna(), '原始序列')
adf_test(ChinaBank['diff_1'].dropna(), '1阶差分序列')
adf_test(ChinaBank['diff_2'].dropna(), '2阶差分序列')

# 4. ACF和PACF分析
fig = plt.figure(figsize=(12, 7))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(train, lags=20, ax=ax1)
ax1.set_title('自相关函数(ACF)')
ax1.xaxis.set_ticks_position('bottom')

ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(train, lags=20, ax=ax2)
ax2.set_title('偏自相关函数(PACF)')
ax2.xaxis.set_ticks_position('bottom')

plt.tight_layout()
plt.show()

# 5. 模型参数选择
# 确定pq的取值范围
p_min, d_min, q_min = 0, 0, 0
p_max, d_max, q_max = 8, 1, 8

# 初始化存储BIC结果的数据框
results_bic = pd.DataFrame(
    index=['AR{}'.format(i) for i in range(p_min, p_max + 1)],
    columns=['MA{}'.format(i) for i in range(q_min, q_max + 1)]
)

# 遍历所有可能的参数组合
for p, d, q in itertools.product(
        range(p_min, p_max + 1),
        range(d_min, d_max + 1),
        range(q_min, q_max + 1)
):
    if p == 0 and d == 0 and q == 0:
        results_bic.loc['AR{}'.format(p), 'MA{}'.format(q)] = np.nan
        continue
    try:
        model = sm.tsa.ARIMA(train, order=(p, d, q))
        results = model.fit()
        results_bic.loc['AR{}'.format(p), 'MA{}'.format(q)] = results.bic
    except:
        continue

# 转换为浮点型并绘制热力图
results_bic = results_bic[results_bic.columns].astype(float)
fig, ax = plt.subplots(figsize=(10, 8))
ax = sns.heatmap(
    results_bic,
    mask=results_bic.isnull(),
    ax=ax,
    annot=True,
    fmt='.2f',
    cmap="Purples"
)
ax.set_title('不同ARIMA(p,d=0,q)模型的BIC值')
plt.tight_layout()
plt.show()

# 使用自动选择功能确认最佳参数
train_results = sm.tsa.arma_order_select_ic(train, ic=['aic', 'bic'], trend='n', max_ar=8, max_ma=8)
print('\nAIC推荐的最佳模型:', train_results.aic_min_order)
print('BIC推荐的最佳模型:', train_results.bic_min_order)

# 6. 模型训练
# 使用推荐的最佳参数
p, d, q = 6,0,5
print(f'\n使用最佳参数 (p={p}, d={d}, q={q}) 训练模型...')

# 训练ARIMA模型
model = sm.tsa.ARIMA(train, order=(p, d, q))
results = model.fit()

# 输出模型摘要
print("\n模型训练摘要:")
print(results.summary())

# 7. 模型诊断
# 残差分析
resid = results.resid

# 残差的ACF图
fig, ax = plt.subplots(figsize=(12, 5))
# 只绘制ACF，不返回Figure对象
sm.graphics.tsa.plot_acf(resid, lags=40, ax=ax)
ax.set_title('残差的自相关函数')  # 现在ax是Axes对象，可以设置标题
plt.show()

# 残差的直方图
plt.figure(figsize=(10, 6))
sns.histplot(resid, kde=True)
plt.title('残差的分布')
plt.show()

# 8. 模型评估（在测试集上）
# 预测测试集
test_forecast = results.get_forecast(steps=len(test))
test_pred = test_forecast.predicted_mean
conf_int_test = test_forecast.conf_int()

# 绘制训练集、测试集和预测值
plt.figure(figsize=(14, 7))
plt.plot(train.index, train, label='训练集')
plt.plot(test.index, test, label='测试集', color='green')
plt.plot(test.index, test_pred, label='预测值', color='red', linestyle='--')
plt.fill_between(test.index,
                 conf_int_test.iloc[:, 0],
                 conf_int_test.iloc[:, 1],
                 color='pink',
                 alpha=0.3)
plt.title('测试集预测结果对比')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# 计算评估指标
mae = mean_absolute_error(test, test_pred)
rmse = np.sqrt(mean_squared_error(test, test_pred))
mape = np.mean(np.abs((test - test_pred) / test)) * 100

print("\n测试集评估指标:")
print(f"平均绝对误差 (MAE): {mae:.4f}")
print(f"均方根误差 (RMSE): {rmse:.4f}")
print(f"平均绝对百分比误差 (MAPE): {mape:.2f}%")

# 9. 未来预测
# 设置预测未来的天数
forecast_days = 30

# 生成未来日期索引
last_date = sub.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)

# 进行未来预测
forecast = results.get_forecast(steps=forecast_days)
forecast_values = forecast.predicted_mean
conf_int = forecast.conf_int()

# 创建预测序列
forecast_series = pd.Series(forecast_values.values, index=future_dates, name='Forecast')

# 绘制历史数据与未来预测
plt.figure(figsize=(14, 8))
plt.plot(sub.index, sub, label='历史收盘价', color='blue')
plt.plot(forecast_series.index, forecast_series, label='未来预测', color='red', linestyle='--')
plt.fill_between(forecast_series.index,
                 conf_int.iloc[:, 0],
                 conf_int.iloc[:, 1],
                 color='pink',
                 alpha=0.3,
                 label='95%置信区间')

plt.title('股票收盘价历史数据与未来预测')
plt.xlabel('日期')
plt.ylabel('收盘价')
plt.xticks(rotation=45)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# 输出未来预测结果
print(f"\n未来{forecast_days}天的预测收盘价:")
print(forecast_series.round(2))
