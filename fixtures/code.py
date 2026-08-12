def calculate_repair_cost(area, rate):
    if area <= 0:
        raise ValueError("area must be positive")
    return round(area * rate, 2)
