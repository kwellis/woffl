from ortools.algorithms.python import knapsack_solver
from ortools.sat.python import cp_model


def standard_knapsack():
    """Standard 0/1 Knapsack using the dedicated knapsack solver.

    Each item is independent — take it or leave it.
    """
    print("=== Standard 0/1 Knapsack ===")

    solver = knapsack_solver.KnapsackSolver(
        knapsack_solver.SolverType.KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER,
        "KnapsackExample",
    )

    values = [
        # fmt: off
        360, 83, 59, 130, 431, 67, 230, 52, 93, 125, 670, 892, 600, 38, 48, 147,
        78, 256, 63, 17, 120, 164, 432, 35, 92, 110, 22, 42, 50, 323, 514, 28,
        87, 73, 78, 15, 26, 78, 210, 36, 85, 189, 274, 43, 33, 10, 19, 389, 276,
        312
        # fmt: on
    ]
    weights = [
        # fmt: off
        [7, 0, 30, 22, 80, 94, 11, 81, 70, 64, 59, 18, 0, 36, 3, 8, 15, 42, 9, 0,
         42, 47, 52, 32, 26, 48, 55, 6, 29, 84, 2, 4, 18, 56, 7, 29, 93, 44, 71,
         3, 86, 66, 31, 65, 0, 79, 20, 65, 52, 13],
        # fmt: on
    ]
    capacities = [850]

    solver.init(values, weights, capacities)
    computed_value = solver.solve()

    packed_items = []
    packed_weights = []
    total_weight = 0
    print("Total value =", computed_value)
    for i in range(len(values)):
        if solver.best_solution_contains(i):
            packed_items.append(i)
            packed_weights.append(weights[0][i])
            total_weight += weights[0][i]
    print("Total weight:", total_weight)
    print("Packed items:", packed_items)
    print("Packed_weights:", packed_weights)
    print()


def multiple_choice_knapsack():
    """Multiple-Choice Knapsack using CP-SAT.

    Toy version of the jet pump network problem. 4 wells, each has 3 candidate
    jet pumps (small, medium, large). Each pump has an oil rate (profit) and a
    lift water rate (weight). Must pick exactly one pump per well. Maximize total
    oil subject to a shared power fluid capacity.

    Well 1: small=50 oil/400 water, medium=80 oil/800 water, large=95 oil/1200 water
    Well 2: small=30 oil/300 water, medium=55 oil/700 water, large=70 oil/1100 water
    Well 3: small=60 oil/500 water, medium=90 oil/900 water, large=100 oil/1400 water
    Well 4: small=40 oil/350 water, medium=65 oil/750 water, large=75 oil/1000 water

    Total power fluid capacity: 3000 bwpd
    """
    print("=== Multiple-Choice Knapsack (CP-SAT) ===")

    #               (oil_rate, lift_water)
    wells = {
        "Well_1": [(50, 400), (80, 800), (95, 1200)],
        "Well_2": [(30, 300), (55, 700), (70, 1100)],
        "Well_3": [(60, 500), (90, 900), (100, 1400)],
        "Well_4": [(40, 350), (65, 750), (75, 1000)],
    }
    capacity = 3000  # total power fluid, bwpd

    model = cp_model.CpModel()

    # create a boolean variable for each (well, pump) pair
    x = {}
    for i, (well_name, pumps) in enumerate(wells.items()):
        for j in range(len(pumps)):
            x[i, j] = model.new_bool_var(f"{well_name}_pump_{j}")

        # each well must pick exactly one pump
        model.add_exactly_one(x[i, j] for j in range(len(pumps)))

    # total lift water cannot exceed capacity
    model.add(
        sum(
            x[i, j] * wells[well_name][j][1]
            for i, well_name in enumerate(wells)
            for j in range(len(wells[well_name]))
        )
        <= capacity
    )

    # maximize total oil
    model.maximize(
        sum(
            x[i, j] * wells[well_name][j][0]
            for i, well_name in enumerate(wells)
            for j in range(len(wells[well_name]))
        )
    )

    solver = cp_model.CpSolver()
    status = solver.solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print(f"Total oil: {solver.objective_value:.0f} bopd")
        total_water = 0
        for i, (well_name, pumps) in enumerate(wells.items()):
            for j, (oil, water) in enumerate(pumps):
                if solver.value(x[i, j]):
                    size = ["small", "medium", "large"][j]
                    print(f"  {well_name}: {size} pump  =>  oil={oil} bopd, lift_water={water} bwpd")
                    total_water += water
        print(f"Total lift water: {total_water} / {capacity} bwpd")
    else:
        print("No solution found")
    print()


if __name__ == "__main__":
    standard_knapsack()
    multiple_choice_knapsack()
