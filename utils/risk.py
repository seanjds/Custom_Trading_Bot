# utils/risk.py
def position_size(account_value, risk_per_trade, entry_price, stop_price):
    """
    Returns number of units to buy given dollar risk and stop price.
    dollar_risk = account_value * risk_per_trade
    per_unit_risk = entry_price - stop_price
    size = dollar_risk / per_unit_risk
    """
    dollar_risk = account_value * risk_per_trade
    per_unit_risk = abs(entry_price - stop_price)
    if per_unit_risk == 0:
        return 0
    return dollar_risk / per_unit_risk
