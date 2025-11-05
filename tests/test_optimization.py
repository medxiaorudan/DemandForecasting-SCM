from src.optimization.inventory_optimization import plan_reorder

def test_plan_feasible():
    demand = [20,22,19,23,25,18,20]
    res = plan_reorder(
        demand=demand,
        initial_inventory=50,
        holding_cost=0.1,
        stockout_cost=1.0,
        service_level=0.95,
        max_order=80,
        lead_time=1
    )
    assert len(res["order_plan"]) == len(demand)
    # Service level approx -> backorders should be relatively small vs demand
    assert sum(res["backorder"]) <= 0.2 * sum(demand)
