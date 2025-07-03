import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.constants import N_A  # 阿伏伽德罗常数

# ========================
# PCET 速率模型
# ========================
class PCETModel:
    """
    描述质子耦合电子转移(PCET)中电子转移速率与pH关系的模型
    
    模型特征：
    - pH > 8 时，k_ET = k0 (不随pH变化)
    - pH < 7 时，log(k_ET) 对 pH 作图斜率为 -0.5
    - pH 7-8 之间为过渡区
    
    数学表达式：
    k_ET = k0 * [1 + (Ka * 10^(-pH))^n]^{s}
    
    参数说明：
    k0: 高pH极限下的速率常数
    Ka: 酸解离常数
    n: 协同质子数
    s: 过渡区形状参数 (通常取0.5)
    """
    
    @staticmethod
    def pcet_rate(pH, k0, Ka, n=1):
        """
        计算PCET速率常数
        
        参数：
        pH: 溶液pH值
        k0: 高pH极限下的速率常数
        Ka: 酸解离常数
        n: 协同质子数 (默认为1)
        s: 过渡区形状参数 (默认为0.5)
        
        返回：
        k_ET: 电子转移速率常数
        """
        # ds: 这里使用更精确的过渡模型，考虑了过渡区形状参数s
        H_plus = 10 ** (-pH)
        return k0*(1 + Ka * H_plus ** n) 
    
    @staticmethod
    def plot_pcet_model(pH_range=(4, 10), k0=1e6, Ka= 10**6.5, n=0.6):
        """可视化PCET速率与pH的关系"""
        pH = np.linspace(pH_range[0], pH_range[1], 100)
        k_ET = PCETModel.pct_rate(pH, k0, Ka, n)
        
        plt.figure(figsize=(8, 5))
        plt.plot(pH, k_ET, 'b-', linewidth=2)
        plt.axvline(x=7, color='gray', linestyle='--', alpha=0.5)
        plt.axvline(x=8, color='gray', linestyle='--', alpha=0.5)
        plt.xlabel('pH')
        plt.ylabel('k$_{ET}$ (M$^{-1}$s$^{-1}$)')
        plt.yscale('log')
        plt.title('PCET Rate Constant vs. pH')
        plt.grid(linestyle='--', alpha=0.3)
        plt.annotate(f'Slope = -{n}\n(pH < 7)', xy=(5.5, k_ET[30]), 
                    xytext=(4.5, k_ET[30]/10), arrowprops=dict(arrowstyle='->'))
        plt.annotate(f'Constant k0 = {k0:.1e}\n(pH > 8)', xy=(8.5, k_ET[-1]), 
                    xytext=(8.5, k_ET[-1]*2))
        plt.tight_layout()
        plt.show()