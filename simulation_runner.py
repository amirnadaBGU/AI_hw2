import random

from DCOP import DCOPInstance
from agents import Agent, DSAAgent
from simulation import Simulation, plot_costs

if __name__ == '__main__':
    p1 = 0.75
    k_values = [0.25]
    algorithms = [
        ("DSA", 0.2),
        ("DSA", 0.7),
        ("DSA", 1.0),
        ("MGM", None),
        ("MGM2", None),
    ]
    for k in k_values:
        all_histories = {}
        problem_instances = [DCOPInstance(30, 5, p1, seed=run) for run in range(1)]
        for problem in problem_instances:
            agents = build_agents_from_problem(problem, domain, algorithm, p)
            sim = Simulation(agents, k)
            initial_cost = sim.compute_global_cost()
            sim.history.append(initial_cost)  # Add pre-run global cost
            sim.run()
            histories.append(sim.history)

        for alg in algorithms:
            history[alg] = []
            sim = Simulation(DCOP, alg, p_dsa=0.7)
            sim.run(steps=125)
            history[alg][0] = sim.history
        plot_costs(history, p1)



        # For graph coloring- restrict domain to color keys
        if k == 0.1:
            color_map = {0: "yellow", 1: "red", 2: "blue"}
            domain = list(color_map.keys())
            coloring = True
        else:
            domain = [0, 1, 2, 3, 4]
            coloring = False



        # Create consistent problem instances across runs

        for algorithm, p in algorithms:
            histories = []
            for problem in problem_instances:
                agents = build_agents_from_problem(problem, domain, algorithm, p)
                sim = Simulation(agents, k)
                initial_cost = sim.compute_global_cost()
                sim.history.append(initial_cost)  # Add pre-run global cost
                sim.run()
                histories.append(sim.history)

            # Compute average history across runs
            avg_history = []
            max_len = max(len(h) for h in histories)
            for t in range(max_len):
                values_at_t = [h[t] if t < len(h) else h[-1] for h in histories]
                avg_history.append(sum(values_at_t) / len(values_at_t))

            all_histories.setdefault(algorithm, {})[p] = avg_history

        plot_costs(all_histories, k)