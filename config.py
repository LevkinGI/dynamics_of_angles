from dataclasses import dataclass

@dataclass
class SimParams:
    alpha_scale: float   # множитель для коэффициента затухания α
    k_scale:  float  # множитель для K(T)
    m_scale: float  # множитель массива m
    M_scale: float  # множитель массива M
