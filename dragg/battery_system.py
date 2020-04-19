class BatterySystem:
    """Battery system model.  System model and constraints from: http://dx.doi.org/10.1016/j.apenergy.2017.08.166"""

    def __init__(self, charge_efficiency, discharge_efficiency, capacity, charge_rate, discharge_rate):
        """
        init
        :param charge_efficiency: float, battery charging efficiency (0 <= eff <= 1)
        :param discharge_efficiency: float, battery discharging efficiency (0 <= eff <= 1)
        :param capacity: float, capacity of the battery (kWh)
        :param charge_rate: float, battery charging rate (kW)
        :param discharge_rate: float, battery discharging rate (kW)
        """
        self.charge_efficiency = charge_efficiency
        self.discharge_efficiency = discharge_efficiency
        self.capacity = capacity
        self.charge_rate = charge_rate
        self.discharge_rate = discharge_rate
