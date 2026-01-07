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
        for algorithm, p in algorithms:
            histories = []
            for DCOP in problem_instances:
                Sim = Simulation(DCOP, algorithm, p_dsa=p)
                Sim.run(steps=125)
                histories.append(Sim.history)
            # Compute average history across runs
            avg_history = []
            max_len = max(len(h) for h in histories)
            for t in range(max_len):
                values_at_t = [h[t] if t < len(h) else h[-1] for h in histories]
                avg_history.append(sum(values_at_t) / len(values_at_t))

            all_histories.setdefault(algorithm, {})[p] = avg_history

        plot_costs(all_histories, k)



