import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad  # 用于计算定积分
import matplotlib.pyplot as plt
import math

# 设置中文字体支持
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False

# 定义常量（同原代码）
pi = math.pi
y1, y2, y3,yw,= 24, 18,10.2,9.8
B1, B2, b1 = 2, 1, 0.3
H, t, L, d1 = 6, 0.4, 2.5, 0.3
n, D, r = 3, 3, 0.4
φ, bt1 = pi / 6, 0
tt, tf, u = 0.92 * 1000000, 300000, 0.6
ffy = 2.9 * 1000000
d = 0.08
A = pi * d * d / 4
α, β = pi * 3 / 4, pi * 2 / 3
ni=4

# 存储迭代过程数据
iterations = []
objective_values = []  # 存储包含积分的kw值
variable_values = []


# --------------------------
# 定义定积分的被积函数（示例）
# 被积函数可以依赖优化变量x
# --------------------------
def integrand(t, x):
    """
    定积分的被积函数
    t: 积分变量
    x: 优化变量（x[0], x[1], x[2]）
    """
    # 示例：被积函数 = (t^2 + x[0]) * exp(-x[1]*t) + x[2]
    # 这里的形式可以根据实际问题修改
    rr=x[0]
    nn=int(x[1])
    kk=x[2:2+nn]
    h=[0,0,0,0]
    s=0
    sig=0
    s=pi*rr*rr
    h[0]=kk[0]
    for i in range(1,nn):
        h[i]=kk[i]+h[i-1]
    for i in range(nn):
        sig+=s*nn/(L*H/math.sin(β)*0.02)*(t-h[i])
    return yw*(t-sig)


# --------------------------
# 包含定积分的目标函数
# --------------------------
def objective(x):
    # 1. 计算原代码中的基础项（保持不变）
    integral_lower = 0  # 积分下限
    integral_upper = H  # 积分上限
    b2 = b1 - H / math.tan(α) - H / math.tan(β)
    C = y1 * (H * B1 * t / 2 + (B1 + B2 + b2) * d1 * (4 * L + 3 * t) + 1 / 2 * (b1 + b2) * (
            4 * L + 3 * t) * H + n * D * math.pi * r ** 2)

    EA = 1 / 2 * y2 * H ** 2 * (math.cos(φ - (β - math.pi / 2))) ** 2 / (
            (math.cos((β - math.pi / 2))) ** 2 * math.cos((β - pi / 2) + φ / 3) * (1 + math.sqrt(
        (math.sin(4 * φ / 3) * math.sin(φ - bt1)) / (
                math.cos((β - pi / 2) + φ / 3) * math.cos((β - pi / 2) - bt1)))) ** 2)

    EB = 1 / 2 * y2 * H ** 2 * (math.cos(φ - (α - pi / 2))) ** 2 / (
            (math.cos(α - pi / 2)) ** 2 * math.cos((α - pi / 2) + φ / 3) * (1 + math.sqrt(
        (math.sin(4 * φ / 3) * math.sin(φ - bt1)) / (
                math.cos((α - math.pi / 2) + φ / 3) * math.cos((α - pi / 2) - bt1)))) ** 2)
    integral_result, integral_error = quad(integrand, integral_lower, integral_upper, args=(x,))
    Pw=integral_result*(4 * L + 3 * t)
    function1 = (u * (
            C + EA * math.sin(β - math.pi / 2)  * (4 * L + 3 * t)+ EB * math.sin(
        α - math.pi / 2) * (4 * L + 3 * t)+ Pw * math.sin(α - math.pi / 2) * (4 * L + 3 * t)) + n * math.pi * r * D * tf + EB * math.cos(
        α - math.pi / 2) ) / (EA * math.cos(β - math.pi / 2) * (4 * L + 3 * t)+Pw*math.cos(β - math.pi / 2) * (4 * L + 3 * t))

    GG = 3 * y1 * B1 * H * t / 2 * ((2 * (- H / math.tan(β)) + B1) / 3 + b1 - H / math.tan(α) + B2) + y1 * (
            B1 + B2 + b2) * (4 * L + 3 * t) * d1 * (B1 + B2 + b2) / 2 + (
                 (2 * b1 + b2) * (- H / math.tan(α)) + (b1 + 2 * b2) * (- H / math.tan(α) + b1) / 3 / (
                 b1 + b2) + B2) * H * (b1 + b2) * (4 * L + 3 * t) / 2 + n * D * pi * r ** 2 * y1 * (B2 + b2 / 2)

    function2 = (GG + (B2 + b2 + H / (3 * math.tan(β))) * EA * math.sin(β - math.pi / 2) + (
            B2 - H / (3 * math.tan(α))) * EB * math.sin(α - pi / 2) + n * math.pi * r * D * tf * D / 2 ) / (
                        (EA * math.cos(β - math.pi / 2)* H / 3 - EB * math.cos(α - math.pi / 2)) * H / 3 + Pw * math.cos(β - math.pi / 2)* H / 3)

    # 2. 计算定积分（核心修改：添加积分项）
    # 积分上下限可以是常数，也可以依赖x（示例中为0到10）
    integral_lower = 0  # 积分下限
    integral_upper = H  # 积分上限
    # 调用quad计算定积分，args=(x,)表示将x作为额外参数传入被积函数
    integral_result, integral_error = quad(integrand, integral_lower, integral_upper, args=(x,))
    # integral_error是积分的数值误差，通常可忽略

    # 3. 定义包含积分的目标函数kw（示例：原kw + 积分结果的加权值）
    # 具体组合方式根据实际问题调整
    kw = 0.5 * (function1 + function2)

    # 记录迭代数据
    current_iter = len(iterations)
    iterations.append(current_iter)
    objective_values.append(kw)  # 存储包含积分的kw值
    variable_values.append(x.copy())

    return -kw  # 最小化负的kw（等价于最大化kw）


# 约束条件（同原代码）
def constraint1(x):
    rr = x[0]
    nn = int(x[1])
    kk = x[2:2 + nn]
    h = [0, 0, 0, 0]
    s = 0
    sig = 0
    s = pi * rr * rr
    h[0] = kk[0]
    if h[0]<rr+d1:
        return h[0]-rr-d1
    for i in range(1, nn):
        h[i] = kk[i] + h[i - 1]
        if h[i]<rr:
            return h[i]-rr
        if h[i]>H:
            return H-h[i]
    return (L*H/math.sin(β)*0.02)-s*nn


# 初始猜测值、约束和边界（同原代码）
x0 = [0.166,4.2,3,1.6,1.6,1.6]

constraints = [
    {'type': 'ineq', 'fun': constraint1}
]

bounds_constraints = [
    {'type': 'ineq', 'fun': lambda x: x[0] - 0.05},
    {'type': 'ineq', 'fun': lambda x: math.sqrt((L*H/math.sin(β)*0.02)/math.pi/x[1]) - x[0]},
    {'type': 'ineq', 'fun': lambda x: x[1] -1},
    {'type': 'ineq', 'fun': lambda x: 4.9 - x[1]},
    {'type': 'ineq', 'fun': lambda x: x[3] - 1.5},
    {'type': 'ineq', 'fun': lambda x: H/int(x[1]) - x[3]},
    {'type': 'ineq', 'fun': lambda x: x[4] - 1.5},
    {'type': 'ineq', 'fun': lambda x: H / int(x[1]) - x[4]},
    {'type': 'ineq', 'fun': lambda x: x[5] - 1.5},
    {'type': 'ineq', 'fun': lambda x: H / int(x[1]) - x[5]},
    {'type': 'ineq', 'fun': lambda x: x[2] - 0},
    {'type': 'ineq', 'fun': lambda x: H / int(x[1]) - x[2]}
]

all_constraints = constraints + bounds_constraints

# 求解优化问题
result = minimize(
    objective,
    x0,
    method='COBYLA',
    constraints=all_constraints,
    options={
        'disp': True,
        'maxiter': 10000000,
        'rhobeg': 0.1,
        'tol': 1e-6
    }
)

# 输出结果
if result.success:
    optimal_x = result.x
    optimal_value = -result.fun
    print("优化成功！")
    print(f"最优解：R = {optimal_x[0]:.4f}, m = {optimal_x[1]:.4f}")
    x=optimal_x
    rr = x[0]
    nn = int(x[1])
    kk = x[2:2 + nn]
    h = [0, 0, 0, 0]
    s = 0
    sig = 0
    s = pi * rr * rr
    h[0] = kk[0]
    print(h[0])
    for i in range(1, nn):
        h[i] = kk[i] + h[i - 1]
        print(h[i])
    print(f"最优值：kw = {optimal_value:.4f}")
else:
    print("优化失败，原因：", result.message)
    print("当前解：X = {0:.4f}, Y = {1:.4f}, Z = {2:.4f}".format(*result.x))
    print("当前目标函数值：kw = {0:.4f}".format(-result.fun))

# 绘制迭代图
plt.figure(figsize=(15, 10))

# 目标函数值（含积分）收敛过程
plt.subplot(1, 1, 1)
plt.plot(iterations, objective_values, 'b-', linewidth=2)
plt.xlabel('迭代次数')
plt.ylabel('目标函数值 (kw)')
plt.title('含定积分的目标函数收敛过程')
plt.grid(True)
if result.success:
    plt.axhline(y=optimal_value, color='r', linestyle='--', label=f'最优值: {optimal_value:.4f}')
    plt.legend()
plt.tight_layout()
plt.show()