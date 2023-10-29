from pint import UnitRegistry

ureg = UnitRegistry()
Q_ = ureg.Quantity

standard_temperature = Q_(298.15, ureg.K)
