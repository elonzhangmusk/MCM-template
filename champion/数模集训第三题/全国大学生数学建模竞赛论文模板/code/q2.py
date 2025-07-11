import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import fsolve
# 设置字体以支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def calculate_partial_derivatives(function_expr, variables):
    symbols = sp.symbols(variables)
    derivatives = {}
    for var in symbols:
        derivatives[str(var)] = sp.diff(function_expr, var)
    return derivatives


def export_to_latex(function, derivatives, vars_list, filename="derivatives.tex"):
    with open(filename, "w") as f:
        f.write(r"\documentclass{article}" + "\n")
        f.write(r"\usepackage{amsmath,amssymb}" + "\n")
        f.write(r"\begin{document}" + "\n\n")
        f.write(r"\section*{原函数}" + "\n")
        f.write(r"$$f\left(" + ", ".join(vars_list) + r"\right) =" + "\n")
        f.write(sp.latex(function) + "$$\n\n")
        f.write(r"\section*{偏导数}" + "\n")
        for var, deriv in derivatives.items():
            f.write(r"$$\frac{\partial f}{\partial " + var + "} =" + "\n")
            f.write(sp.latex(deriv) + "$$\n\n")
        f.write(r"\end{document}" + "\n")
    print(f"LaTeX文档已保存至 {filename}")


def plot_function_and_derivatives(function, derivatives, vars_list, param_values,
                                  var_ranges=None, save_figures=True):
    if var_ranges is None:
        var_ranges = {
            vars_list[0]: (0, 2 * np.pi, 100),
            vars_list[1]: (0, 2 * np.pi, 100)
        }
    var1, var2 = vars_list
    x_min, x_max, x_num = var_ranges[var1]
    y_min, y_max, y_num = var_ranges[var2]
    x = np.linspace(x_min, x_max, x_num)
    y = np.linspace(y_min, y_max, y_num)
    X, Y = np.meshgrid(x, y)

    func_lambda = sp.lambdify((sp.symbols(var1), sp.symbols(var2)),
                              function.subs(param_values), "numpy")
    deriv_lambdas = {
        var: sp.lambdify((sp.symbols(var1), sp.symbols(var2)),
                         deriv.subs(param_values), "numpy")
        for var, deriv in derivatives.items()
    }

    try:
        Z = func_lambda(X, Y)
    except Exception as e:
        print(f"计算原函数时出错: {e}")
        Z = np.zeros_like(X)

    Z_derivs = {}
    for var, deriv_lambda in deriv_lambdas.items():
        try:
            Z_derivs[var] = deriv_lambda(X, Y)
        except Exception as e:
            print(f"计算偏导数 ∂f/∂{var} 时出错: {e}")
            Z_derivs[var] = np.zeros_like(X)

    plt.figure(figsize=(15, 10))
    ax1 = plt.subplot(2, 2, 1, projection='3d')
    surf1 = ax1.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=True)
    ax1.set_title('原函数')
    ax1.set_xlabel(var1)
    ax1.set_ylabel(var2)
    ax1.set_zlabel('f')
    plt.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

    for i, (var, Z_deriv) in enumerate(Z_derivs.items(), 2):
        ax = plt.subplot(2, 2, i, projection='3d')
        surf = ax.plot_surface(X, Y, Z_deriv, cmap=cm.coolwarm, linewidth=0, antialiased=True)
        ax.set_title(f'$\sigma$f/$\sigma${var}')
        ax.set_xlabel(var1)
        ax.set_ylabel(var2)
        ax.set_zlabel(f'$\sigma$f/$\sigma${var}')
        plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    plt.tight_layout()
    if save_figures:
        plt.savefig('function_and_derivatives.png', dpi=300, bbox_inches='tight')
        print("图像已保存为 function_and_derivatives.png")
    plt.show()


def find_partial_derivative_zeros(derivatives, vars_list, param_values, var_ranges, method='numerical'):
    symbols = sp.symbols(vars_list)
    zeros = {}

    for var, deriv in derivatives.items():
        print(f"\n求解 ∂f/∂{var} = 0 ...")
        deriv_substituted = deriv.subs(param_values)
        equation = sp.Eq(deriv_substituted, 0)

        if method == 'symbolic':
            try:
                sol = sp.solve(equation, sp.Symbol(var), dict=True)
                zeros[var] = sol
                print(f"∂f/∂{var} 的符号解：")
                for s in sol:
                    sp.pprint(s)
            except Exception as e:
                print(f"符号求解失败：{e}，自动切换为数值求解")
                method = 'numerical'

        if method == 'numerical':
            var_sym = sp.Symbol(var)
            other_var = [v for v in vars_list if v != var][0]
            other_var_sym = sp.Symbol(other_var)
            other_var_mid = (var_ranges[other_var][0] + var_ranges[other_var][1]) / 2

            deriv_num = sp.lambdify(
                var_sym,
                deriv_substituted.subs({other_var_sym: other_var_mid}),
                'numpy'
            )

            var_min, var_max, _ = var_ranges[var]
            search_points = np.linspace(var_min, var_max, 50)
            roots = []

            for x0 in search_points:
                try:
                    root = fsolve(deriv_num, x0)[0]
                    if (var_min <= root <= var_max) and np.isclose(deriv_num(root), 0, atol=1e-3):
                        if not any(np.isclose(root, r, atol=1e-3) for r in roots):
                            roots.append(round(root, 6))
                except:
                    continue

            zeros[var] = roots
            print(f"∂f/∂{var} 的数值解（在范围内）：{roots}")
            print(f"验证：∂f/∂{var} 在零点处的值：{[round(deriv_num(r), 9) for r in roots]}")

    return zeros


# 新增：绘制偏导数图像，辅助分析零点
def plot_partial_derivatives(derivatives, vars_list, param_values, var_ranges):
    """绘制偏导数函数图像，辅助分析零点位置"""
    for var, deriv in derivatives.items():
        var_sym = sp.Symbol(var)
        other_var = [v for v in vars_list if v != var][0]
        other_var_sym = sp.Symbol(other_var)
        other_var_mid = (var_ranges[other_var][0] + var_ranges[other_var][1]) / 2

        # 转换为数值函数
        deriv_num = sp.lambdify(
            var_sym,
            deriv.subs({**param_values, other_var_sym: other_var_mid}),
            'numpy'
        )

        # 创建绘图数据
        var_min, var_max, var_num = var_ranges[var]
        x = np.linspace(var_min, var_max, var_num)
        y = deriv_num(x)

        # 绘制函数
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, label=f'$\sigma$f/$\sigma${var}')
        plt.axhline(y=0, color='r', linestyle='--', label='y=0')  # 添加y=0参考线
        plt.title(f'偏导数 $\sigma$f/$\sigma${var} 图像')
        plt.xlabel(var)
        plt.ylabel(f'∂f/∂{var}')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'partial_derivative_{var}.png')
        plt.show()

        # 检查是否存在接近零的值
        close_to_zero = np.isclose(y, 0, atol=1e-3)
        if any(close_to_zero):
            zeros = x[close_to_zero]
            print(f"在 {var} 范围内发现接近零的点：{zeros}")
            print(f"对应的函数值：{y[close_to_zero]}")
        else:
            print(f"在 {var} 范围内未发现接近零的点")


# 新增：寻找函数的驻点（所有偏导数同时为零的点）
def find_critical_points(function, vars_list, param_values, var_ranges):
    """寻找函数的驻点（所有偏导数同时为零的点）"""
    symbols = sp.symbols(vars_list)

    # 计算所有偏导数
    partials = [sp.diff(function, s) for s in symbols]

    # 创建多变量数值函数
    def equations(x):
        # x 是包含所有变量值的数组
        subs_dict = {**param_values}
        for i, s in enumerate(symbols):
            subs_dict[s] = x[i]

        # 计算所有偏导数在给定点的值
        return [float(p.subs(subs_dict)) for p in partials]

    # 在变量范围内均匀采样初始点
    var1_min, var1_max, _ = var_ranges[vars_list[0]]
    var2_min, var2_max, _ = var_ranges[vars_list[1]]

    # 尝试多个初始点
    critical_points = []
    for x1 in np.linspace(var1_min, var1_max, 5):
        for x2 in np.linspace(var2_min, var2_max, 5):
            try:
                result = fsolve(equations, [x1, x2])
                # 验证解是否在范围内且满足方程
                if (var1_min <= result[0] <= var1_max) and (var2_min <= result[1] <= var2_max):
                    if np.allclose(equations(result), [0, 0], atol=1e-3):
                        # 去重
                        if not any(np.allclose(result, cp, atol=1e-3) for cp in critical_points):
                            critical_points.append(result)
                            print(f"找到驻点：{result}, 偏导数的值：{equations(result)}")
            except:
                continue

    return critical_points


if __name__ == "__main__":
    α, φ, y1, y2, H, B1, B2, b1, b2, L, t, d1, u, β, bt1, n, D, tf, r = sp.symbols(
        'α φ y1 y2 H B1 B2 b1 b2 L t d1 u β bt1 n D tf r'
    )

    # 定义原函数
    C = y1 * (H * B1 * t / 2 + (B1 + B2 + b2) * d1 * (4 * L + 3 * t) + 1 / 2 * (b1 + b2) * (
                4 * L + 3 * t) * H + n * D * sp.pi * r ** 2)
    EA = 1 / 2 * y2 * H ** 2 * (4 * L + 3 * t) * (sp.cos(φ - (β - sp.pi / 2))) ** 2 / (
            (sp.cos((β - sp.pi / 2))) ** 2 * sp.cos((β - sp.pi / 2) + φ / 3) * (1 + sp.sqrt(
        (sp.sin(4 * φ / 3) * sp.sin(φ - bt1)) / (
                    sp.cos((β - sp.pi / 2) + φ / 3) * sp.cos((β - sp.pi / 2) - bt1)))) ** 2)
    EB = 1 / 2 * y2 * H ** 2 * (4 * L + 3 * t) * (sp.cos(φ - (α - sp.pi / 2))) ** 2 / (
            (sp.cos(α - sp.pi / 2)) ** 2 * sp.cos((α - sp.pi / 2) + φ / 3) * (1 + sp.sqrt(
        (sp.sin(4 * φ / 3) * sp.sin(φ - bt1)) / (
                sp.cos((α - sp.pi / 2) + φ / 3) * sp.cos((α - sp.pi / 2) - bt1)))) ** 2)
    function1 = (u * (
                C + EA * sp.sin(β - sp.pi / 2) + EB * sp.sin(α - sp.pi / 2)) + n * sp.pi * r * D * tf + EB * sp.cos(
        α - sp.pi / 2)) / (EA * sp.cos(β - sp.pi / 2))
    GG = 3 * y1 * B1 * H * t / 2 * ((2 * (- H / sp.tan(β)) + B1) / 3 + b1 - H / sp.tan(α) + B2) + y1 * (
                B1 + B2 + b2) * (4 * L + 3 * t) * d1 * (B1 + B2 + b2) / 2 + (
                 (2 * b1 + b2) * (- H / sp.tan(α)) + (b1 + 2 * b2) * (- H / sp.tan(α) + b1) / 3 / (
                     b1 + b2) + B2) * H * (b1 + b2) * (4 * L + 3 * t) / 2 + n * D * sp.pi * r ** 2 * y1 * (B2 + b2 / 2)
    function2 = (GG + (B2 + b2 + H / (3 * sp.tan(β))) * EA * sp.sin(β - sp.pi / 2) + (
            B2 - H / (3 * sp.tan(α))) * EB * sp.sin(α - sp.pi / 2) + n * sp.pi * r * D * tf * D / 2) / (
                        (EA * sp.cos(β - sp.pi / 2) + EB * sp.cos(α - sp.pi / 2)) * H / 3)
    function = (function1 + function2) / 2
    vars_list = ['α', 'β']
    partials = calculate_partial_derivatives(function, vars_list)

    # 输出原函数和偏导数
    print(f"原函数: f({', '.join(vars_list)}) =")
    sp.pprint(function)
    for var, deriv in partials.items():
        print(f"\n$\sigma$f/$\sigma${var} =")
        sp.pprint(deriv)

    # 导出为LaTeX
    export_to_latex(function, partials, vars_list)

    # 设置参数值和变量范围
    param_values = {
        y1: 24.0, y2: 18.0, H: 10.0, B1: 3.0, B2: 1.5,
        b1: 0.3, b2: 2.5, L: 4.0, t: 0.5,
        d1: 0.4, u: 0.6,tf:300000,
        bt1: 0, φ:sp.pi/6, r:0.4,
        n:3,D:3
    }
    hk = 0.001
    var_ranges = {
        'α': (np.pi / 2, 2.8, 150),
        'β': (np.pi / 2, 2.8, 150)
    }

    # 绘制函数图像
    plot_function_and_derivatives(function, partials, vars_list, param_values, var_ranges)

    # 绘制偏导数图像，辅助分析零点
    plot_partial_derivatives(partials, vars_list, param_values, var_ranges)

    # 尝试寻找驻点
    critical_points = find_critical_points(function, vars_list, param_values, var_ranges)
    if critical_points:
        print(f"\n找到 {len(critical_points)} 个驻点:")
        for i, cp in enumerate(critical_points):
            print(f"驻点 {i + 1}: {vars_list[0]} = {cp[0]:.6f}, {vars_list[1]} = {cp[1]:.6f}")
    else:
        print("\n在指定范围内未找到驻点")

    # 尝试扩展变量范围
    expanded_var_ranges = {
        'α': (np.pi / 4, np.pi * 7 / 8, 200),  # 扩大af范围
        'β': (-np.pi / 4, np.pi / 2, 200)  # 扩大fy范围并包含负值
    }

    print("\n尝试在扩展范围内寻找偏导数零点...")
    expanded_zeros = find_partial_derivative_zeros(
        derivatives=partials,
        vars_list=vars_list,
        param_values=param_values,
        var_ranges=expanded_var_ranges,
        method='numerical'
    )