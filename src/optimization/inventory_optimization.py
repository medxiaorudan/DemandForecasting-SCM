from typing import List, Optional, Dict
from ortools.linear_solver import pywraplp

def plan_reorder(
    demand: List[float],
    initial_inventory: int,
    holding_cost: float = 0.1,
    stockout_cost: float = 1.0,
    service_level: float = 0.95,
    max_order: Optional[int] = None,
    lead_time: int = 0
) -> Dict:
    # Single-item planning:
    #   - Decision: order_t >= 0 integer
    #   - Inventory balance with lead time
    #   - Backorders penalized by stockout_cost
    #   - Approx service level via total backorder <= (1 - service_level) * total demand
    T = len(demand)
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if solver is None:
        raise RuntimeError("OR-Tools solver not available")

    # Variables
    order = [solver.IntVar(0.0, solver.infinity(), f"order_{t}") for t in range(T)]
    inventory = [solver.NumVar(0.0, solver.infinity(), f"inv_{t}") for t in range(T)]
    backorder = [solver.NumVar(0.0, solver.infinity(), f"bo_{t}") for t in range(T)]

    # Inventory balance (with lead time)
    for t in range(T):
        arrivals = order[t - lead_time] if (t - lead_time) >= 0 else 0.0
        prev_inv = inventory[t-1] if t > 0 else initial_inventory
        prev_bo = backorder[t-1] if t > 0 else 0.0
        solver.Add(prev_inv + arrivals - demand[t] + prev_bo == inventory[t] - backorder[t])

    # Max order cap
    if max_order is not None:
        for t in range(T):
            solver.Add(order[t] <= max_order)

    # Service level (approx): total backorders <= (1 - SL) * total demand
    solver.Add(sum(backorder) <= (1 - service_level) * sum(demand))

    # Objective: holding + stockout + small ordering regularization
    solver.Minimize(
        holding_cost * sum(inventory) +
        stockout_cost * sum(backorder) +
        0.001 * sum(order)
    )

    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        raise RuntimeError("No feasible solution found")

    order_plan = [int(round(v.solution_value())) for v in order]
    inv = [float(v.solution_value()) for v in inventory]
    bo = [float(v.solution_value()) for v in backorder]
    total_cost = holding_cost*sum(inv) + stockout_cost*sum(bo) + 0.001*sum(order_plan)
    return {
        "order_plan": order_plan,
        "inventory": inv,
        "backorder": bo,
        "total_cost": total_cost
    }
