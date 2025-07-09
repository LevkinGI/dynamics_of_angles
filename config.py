from dataclasses import dataclass

@dataclass
class SimParams:
    alpha: float   # коэффициент затухания
    chi:   float   # магнитная восприимчивость (теперь скаляр)
    k_scale:  float  # множитель для K(T)
    m_scale: float  # множитель массива m
    M_scale: float  # множитель массива M
