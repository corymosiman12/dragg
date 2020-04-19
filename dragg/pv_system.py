class PVSystem:
    """
    PV system model.  Simplified model where the power generated at time t (Pgen) is only a function of the area, efficiency, and global horizontal irradiance (GHI).
    Pgen(t) = GHI(t) * A * efficiency
    """

    def __init__(self, area, efficiency):
        """
        init
        :param area: float, pv system area (m2)
        :param efficiency: float, pv system efficiency (0 <= eff <= 1)
        """
        self.area = area
        self.efficiency = efficiency
