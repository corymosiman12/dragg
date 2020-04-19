class HomeHVACSystem:
    """Home coupled with HVAC system model.  System model and constraints from: https://ieeexplore.ieee.org/abstract/document/8906600"""

    def __init__(self, resistance, capacitance, power_rating):
        """
        init
        :param resistance: float, thermal resistance of the home, (J/K)
        :param capacitance: float, thermal capacitance of the home (K/W)
        :param power_rating: float, power rating of the HVAC in the home (kW)
        """
        self.resistance = resistance
        self.capacitance = capacitance
        self.power_rating = power_rating
