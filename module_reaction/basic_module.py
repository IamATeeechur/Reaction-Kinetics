import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.constants import N_A  # 阿伏伽德罗常数

# ========================
# 物理常数和基础函数模块
# ========================
class Photophysics:
    """处理光物理计算的基础类"""
    
    @staticmethod
    def _non_negative(y):
        """确保所有浓度非负"""
        return np.maximum(y, 0)
    
    @staticmethod
    def calculate_excitation_rate(wav, power, A):
        """
        根据波长、功率等计算出稳态激发条件下，激发生成Ru*的速率
        
        参数：
        wav: 激发光波长 (nm)
        power: 在检测面积上的激发光功率 (mW)
        A: 系统在激发波长上的Abs
        
        输出：
        I_exc：激发态生成速率 (M·s⁻¹)
        """
        # 光子能量 (eV)
        hv = 1240 / wav
        # 光子能量 (J)
        hv = hv * 1.6022e-19
        # 光子通量 (photon/s)
        flux = power / 1000 / hv
        # 吸收光子数
        flux_absorbed = flux * (1 - 10 ** (-A))
        # 假设这里的激发面积是标准的 1cm^2, 比色皿光程 1cm，则激发态空间体积为 1 cm^3 即10^-3 L
        I_exc = flux_absorbed * 1000 / N_A
        return I_exc
    
    @staticmethod
    def calculate_excitation_pulse(wav, pulse_energy, A):
        """
        根据波长、脉冲能量等计算出脉冲激发条件下，每束脉冲激发生成Ru*的浓度
        
        参数：
        wav: 激发光波长 (nm)
        pulse_energy: 在检测面积上的脉冲能量 (mJ)
        A: 系统在激发波长上的Abs
        
        输出：
        I_exc：激发态生成量 (M)
        """
        # 光子能量 (eV)
        hv = 1240 / wav
        # 光子能量 (J)
        hv = hv * 1.6022e-19
        # 光子数
        N_photon = pulse_energy / 1000 / hv
        # 吸收光子数
        photon_absorbed = N_photon * (1 - 10 ** (-A))
        # 假设这里的激发面积是标准的 1cm^2, 比色皿光程 1cm，则激发态空间体积为 1 cm^3 即10^-3 L
        I_exc = photon_absorbed * 1000 / N_A
        return I_exc