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

# 反应动力学系统
# ========================
class ReactionSystem:
    """光化学反应动力学系统"""
    
    def __init__(self, concentrations, epsilon, excitation_params):
        """
        初始化反应系统
        
        参数：
        concentrations: 初始浓度 [Ru, SA, CII] (M)
        epsilon: 摩尔消光系数 (M⁻¹cm⁻¹)
        excitation_params: 激发参数字典
            - 脉冲激发: {'type': 'pulse', 'wavelength': nm, 'pulse_energy': mJ}
            - 稳态激发: {'type': 'steady', 'wavelength': nm, 'power': mW}
        """
        self.Ru_0, self.SA_0, self.C_II_0 = concentrations
        self.epsilon = epsilon
        self.excitation_params = excitation_params
        self.solution = None
        # 设置初始条件
        self._set_initial_conditions()
    
    def _set_initial_conditions(self):
        """根据激发类型设置初始条件"""
        exc_type = self.excitation_params['type']
        
        if exc_type == 'pulse':
            wav = self.excitation_params['wavelength']
            pulse_energy = self.excitation_params['pulse_energy']
            A = self.Ru_0 * self.epsilon
            
            Ru_star0 = Photophysics.calculate_excitation_pulse(wav, pulse_energy, A)
            Ru_star0 = min(Ru_star0, self.Ru_0)  # 确保不超过可用Ru的量
            
            # 初始条件: [Ru, Ru*, SA, SA_r, Ru_I, C_II, Ru_II_C_I, C_0]
            self.y0 = [
                self.Ru_0 - Ru_star0,  # Ru
                Ru_star0,               # Ru*
                self.SA_0,              # SA
                0,                      # SA_r
                0,                      # Ru_I
                self.C_II_0,            # C_II
                0,                      # Ru_II_C_I
                0                       # C_0
            ]
            
        elif exc_type == 'steady':
            # 稳态激发初始条件
            self.y0 = [
                self.Ru_0,  # Ru
                0,         # Ru*
                self.SA_0, # SA
                0,         # SA_r
                0,         # Ru_I
                self.C_II_0, # C_II
                0,         # Ru_II_C_I
                0          # C_0
            ]
        else:
            raise ValueError("Invalid excitation type. Use 'pulse' or 'steady'")

    def ReactionPrint(self):
        print(self.Ru_0, "from the class Reactionsystem and the subfunction")

    def pulse_excitation(self, t, y, k_params):
        """
        脉冲激发下的反应动力学模型
        
        参数：
        t: 时间
        y: 状态变量 [Ru, Ru*, SA, SA_r, Ru_I, C_II, Ru_II_C_I, C_0]
        k_params: 动力学参数字典
        
        微分方程组描述：
        k1: Ru* + SA → RuI + SAR
        不确定的反应机制部分1：k2、k3，RuI的衰减是一级反应还是二级反应
        k2: RuI → Ru
        k3: RuI + SAr → Ru + SA
        k4: RuI + CII → RuIICI (PCET步骤，与pH相关)
        k5: RuIICI → Ru + CII
        不确定的反应机制部分2：k6、k7，第二步电子转移的电子供体是RuI还是SA
        k6: RuIICI + RuI → 2Ru + C0
        k7: RuIICI + SA → Ru + SAr + C0
        不确定的反应机制部分3：k5、k8，烯基自由基的衰减产物
        k8: RuIICI → RuI + CII
        """
        # 确保浓度非负
        y = Photophysics._non_negative(y)
        Ru, Ru_star, SA, SA_r, Ru_I, C_II, Ru_II_C_I, C_0 = y
        
        # 计算反应速率
        r1 = k_params['k1'] * Ru_star * SA
        r2 = k_params['k2'] * Ru_I
        r3 = k_params['k3'] * Ru_I * SA_r
        r4 = k_params['k4'] * Ru_I * C_II
        r5 = k_params['k5'] * Ru_II_C_I
        r6 = k_params['k6'] * Ru_II_C_I * Ru_I
        r7 = k_params['k7'] * Ru_II_C_I * SA
        r8 = k_params['k8'] * Ru_II_C_I
        
        # 微分方程组
        dRu__dt = r2 + r3 + r5 + 2 * r6 + r7
        dRu_star__dt = -r1
        dSA__dt = -r1 + r3 - r7
        dSA_r__dt = r1 - r3 + r7
        dRu_I__dt = r1 - r2 - r3 - r4 - r6 + r8
        dC_II__dt = -r4 + r5 + r8
        dRu_II_C_I__dt = r4 - r5 - r6 - r7 - r8
        dC_0__dt = r6 + r7
        
        return [dRu__dt, dRu_star__dt, dSA__dt, dSA_r__dt, 
                dRu_I__dt, dC_II__dt, dRu_II_C_I__dt, dC_0__dt]
    
    def steady_state_excitation(self, t, y, k_params):
        """
        稳态激发下的反应动力学模型
        
        参数：
        t: 时间
        y: 状态变量 [Ru, Ru*, SA, SA_r, Ru_I, C_II, Ru_II_C_I, C_0]
        k_params: 动力学参数字典
        """
        # 确保浓度非负
        y = Photophysics._non_negative(y)
        Ru, Ru_star, SA, SA_r, Ru_I, C_II, Ru_II_C_I, C_0 = y
        
        # 计算激发速率 (依赖于Ru的浓度)
        A = Ru * self.epsilon
        excitation_rate = Photophysics.calculate_excitation_rate(
            self.excitation_params['wavelength'],
            self.excitation_params['power'],
            A
        )
        
        # 计算反应速率
        r0 = excitation_rate  # 稳态激发速率
        r1 = k_params['k1'] * Ru_star * SA
        r2 = k_params['k2'] * Ru_I
        r3 = k_params['k3'] * Ru_I * SA_r
        r4 = k_params['k4'] * Ru_I * C_II
        r5 = k_params['k5'] * Ru_II_C_I
        r6 = k_params['k6'] * Ru_II_C_I * Ru_I
        r7 = k_params['k7'] * Ru_II_C_I * SA
        r8 = k_params['k8'] * Ru_II_C_I
        
        # 微分方程组
        dRu__dt = -r0 + r2 + r3 + r5 + 2 * r6 + r7
        dRu_star__dt = r0 - r1
        dSA__dt = -r1 + r3 - r7
        dSA_r__dt = r1 - r3 + r7
        dRu_I__dt = r1 - r2 - r3 - r4 - r6 + r8
        dC_II__dt = -r4 + r5 + r8
        dRu_II_C_I__dt = r4 - r5 - r6 - r7 - r8
        dC_0__dt = r6 + r7 
        
        return [dRu__dt, dRu_star__dt, dSA__dt, dSA_r__dt, 
                dRu_I__dt, dC_II__dt, dRu_II_C_I__dt, dC_0__dt]
    
    def run_simulation(self, k_params, t_span, t_eval=None, method='Radau', rtol=1e-6, atol=1e-12):
        """
        运行动力学模拟
        
        参数：
        k_params: 动力学参数字典
        t_span: 时间范围 (start, end)
        t_eval: 评估时间点数组
        method: 求解器方法
        rtol, atol: 求解器容差
        
        返回：
        solution: scipy ODE 求解结果
        """
        exc_type = self.excitation_params['type']
        
        if exc_type == 'pulse':
            ode_func = lambda t, y: self.pulse_excitation(t, y, k_params)
        elif exc_type == 'steady':
            ode_func = lambda t, y: self.steady_state_excitation(t, y, k_params)
        else:
            raise ValueError("Invalid excitation type. Use 'pulse' or 'steady'")
        
        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], 1000)
        
        # ds: 使用更稳健的求解器配置
        sol = solve_ivp(
            ode_func,
            t_span,
            self.y0,
            t_eval=t_eval,
            method=method,
            rtol=rtol,
            atol=atol,
            vectorized=True
        )
        
        self.solution = sol
        return sol

    def plot_results(self, title='Reaction Kinetics'):
        """绘制浓度随时间变化曲线"""
        if self.solution is None:
            raise RuntimeError("Run simulation first before plotting results")
        
        species = ['Ru', 'Ru*', 'SA', 'SA_r', 'RuI', 'C2H2', 'CH2CH·', 'CH2CH2']
        t = self.solution.t
        y = self.solution.y
        
        plt.figure(figsize=(12, 8))
        
        for i, name in enumerate(species):
            plt.subplot(3, 3, i+1)
            plt.plot(t, y[i])
            plt.title(f'{name} Concentration')
            plt.xlabel('Time (s)')
            plt.ylabel('Concentration (M)')
            plt.grid(alpha=0.3, ls = '--')
        
        plt.tight_layout()
        plt.suptitle(title, fontsize=16, y=1.02)
        plt.show()
    
    def compare_models(self, other_system, title='Model Comparison'):
        """比较两种模型的模拟结果"""
        if self.solution is None or other_system.solution is None:
            raise RuntimeError("Both systems must have run simulations")
        
        species = ['Ru', 'Ru*', 'SA', 'SA_r', 'RuI', 'C2H2', 'CH2CH·', 'CH2CH2']
        t1 = self.solution.t
        y1 = self.solution.y
        t2 = other_system.solution.t
        y2 = other_system.solution.y
        
        plt.figure(figsize=(12, 8))
        
        for i, name in enumerate(species):
            plt.subplot(3, 3, i+1)
            plt.plot(t1, y1[i], label='Model 1')
            plt.plot(t2, y2[i], label='Model 2')
            plt.title(f'{name} Concentration')
            plt.xlabel('Time (s)')
            plt.ylabel('Concentration (M)')
            plt.legend(loc = 'best', frameon = True, shadow = True)
            plt.grid(alpha=0.3, ls = '--')
        
        plt.tight_layout()
        plt.suptitle(title, fontsize=16, y=1.02)
        plt.show()

    

   