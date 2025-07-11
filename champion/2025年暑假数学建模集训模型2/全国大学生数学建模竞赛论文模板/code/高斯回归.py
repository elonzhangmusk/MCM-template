import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
import matplotlib.pyplot as plt
import joblib

# 设置字体以支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def load_and_preprocess_data(data_path):
    """加载数据并进行预处理"""
    data = pd.read_excel(data_path)

    X = data.iloc[:, :-1].values  # 特征
    Y = data.iloc[:, -1].values  # 目标变量

    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 划分训练集和测试集
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_scaled, Y, test_size=0.2, random_state=42
    )

    return X_train, X_test, Y_train, Y_test, scaler, X  # 返回原始X用于演示


def train_gpr_model(X_train, Y_train):
    """训练高斯过程回归模型"""
    kernel = C(1.0, (1e-4, 1e1)) * RBF(1.0, (1e-4, 1e1))

    gpr_model = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=10,
        random_state=42
    )

    gpr_model.fit(X_train, Y_train)

    return gpr_model


def evaluate_model(Y_true, Y_pred, dataset_type="数据集"):
    """评估模型性能"""
    mse = mean_squared_error(Y_true, Y_pred)
    mae = mean_absolute_error(Y_true, Y_pred)
    r2 = r2_score(Y_true, Y_pred)
    evs = explained_variance_score(Y_true, Y_pred)

    print(f"{dataset_type}评估结果:")
    print(f"均方误差(MSE): {mse:.4f}")
    print(f"平均绝对误差(MAE): {mae:.4f}")
    print(f"决定系数(R²): {r2:.4f}")
    print(f"解释方差(EVS): {evs:.4f}\n")

    return mse, mae, r2, evs


def plot_predictions(Y_train, Y_train_pred, Y_test, Y_test_pred):
    """绘制预测值与真实值对比图"""
    plt.figure(figsize=(14, 6))

    # 训练集对比
    plt.subplot(1, 2, 1)
    plt.scatter(Y_train, Y_train_pred, color='blue', alpha=0.6, label='预测值')
    plt.plot([Y_train.min(), Y_train.max()], [Y_train.min(), Y_train.max()], 'r--', lw=2, label='理想线')
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.title('训练集: 预测值 vs 真实值')
    plt.legend()

    # 测试集对比
    plt.subplot(1, 2, 2)
    plt.scatter(Y_test, Y_test_pred, color='green', alpha=0.6, label='预测值')
    plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--', lw=2, label='理想线')
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.title('测试集: 预测值 vs 真实值')
    plt.legend()

    plt.tight_layout()
    plt.savefig('prediction_comparison.png', dpi=300)
    plt.show()


def save_model(model, scaler, model_path='gpr_model.pkl', scaler_path='scaler.pkl'):
    """保存模型和标准化器"""
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"模型已保存至 {model_path}")
    print(f"标准化器已保存至 {scaler_path}")


def load_model(model_path='gpr_model.pkl', scaler_path='scaler.pkl'):
    """加载保存的模型和标准化器"""
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print(f"已从 {model_path} 加载模型")
    print(f"已从 {scaler_path} 加载标准化器")
    return model, scaler


def predict_raw_data(raw_data, model, scaler):
    """
    直接接收原始数据并进行预测
    内部会自动完成标准化处理
    """
    # 将原始数据转换为numpy数组（如果不是的话）
    if not isinstance(raw_data, np.ndarray):
        raw_data = np.array(raw_data)

    # 确保输入是二维数组（样本数 x 特征数）
    if raw_data.ndim == 1:
        raw_data = raw_data.reshape(1, -1)

    # 自动标准化处理
    scaled_data = scaler.transform(raw_data)

    # 预测
    predictions, uncertainties = model.predict(scaled_data, return_std=True)

    return predictions, uncertainties


def main():
    # 1. 加载和预处理数据
    data_path = 'data1_2.xlsx'  # 替换为你的数据路径
    X_train, X_test, Y_train, Y_test, scaler, X_original = load_and_preprocess_data(data_path)

    # 2. 训练模型
    print("开始训练模型...")
    gpr_model = train_gpr_model(X_train, Y_train)
    print("模型训练完成!")
    print(f"优化后的核函数: {gpr_model.kernel_}\n")

    # 3. 模型预测
    Y_train_pred, _ = gpr_model.predict(X_train, return_std=True)
    Y_test_pred, _ = gpr_model.predict(X_test, return_std=True)

    # 4. 模型评估
    evaluate_model(Y_train, Y_train_pred, "训练集")
    evaluate_model(Y_test, Y_test_pred, "测试集")

    # 5. 绘制预测对比图
    plot_predictions(Y_train, Y_train_pred, Y_test, Y_test_pred)

    # 6. 保存模型
    save_model(gpr_model, scaler)


    # 8. 演示如何手动输入原始数据进行预测
    print("===== 手动输入原始数据预测示例 =====")
    # 手动输入新的原始数据（特征数量必须与训练数据一致）
    manual_raw_data = [
        [90],
        [91],
        [92],
        [93],
        [94],
        [95],
        [96],
        [97],
        [98],
        [99]
    ]

    # 直接预测
    manual_predictions, manual_uncertainties = predict_raw_data(manual_raw_data, gpr_model, scaler)

    for i in range(len(manual_predictions)):
        print(f"手动输入样本 {i + 1}:")
        print(f"  原始特征: {manual_raw_data[i]}")
        print(f"  预测值: {manual_predictions[i]:.4f}")
        print(f"  不确定性: {manual_uncertainties[i]:.4f}\n")


if __name__ == "__main__":
    main()
