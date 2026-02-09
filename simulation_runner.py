import random
import numpy as np
from DCOP import DCOPInstance
from agents import Agent, DSAAgent
from simulation import Simulation, plot_costs

if __name__ == '__main__':
    p1 = [0.2] # 0.2, 0.5
    space = 10
    algorithms = [
        ("DSA", 0.7),
        ("MGM", None),
        ("MGM2", None)
    ]

    for p in p1:
        all_histories = {}
        #problem_instances = [DCOPInstance(30, 10, p,1, seed=42)]
        problem_instances = [DCOPInstance(30, 10, p, 1, seed=random.randint(1, 100000)) for run in range(50)]
        for algorithm, pdsa in algorithms:
            print(algorithm)
            histories = []
            problem_num = 1
            for DCOP in problem_instances:
                print("problem:",problem_num)
                problem_num+=1
                Sim = Simulation(DCOP, algorithm, p_dsa=pdsa)
                Sim.run(steps=1000)
                histories.append(Sim.history)
                print(Sim.history[-1])
            # Compute average history across runs
            avg_history = []
            max_len = max(len(h) for h in histories)
            for t in range(max_len):
                values_at_t = [h[t] if t < len(h) else h[-1] for h in histories]
                avg_history.append(sum(values_at_t) / len(values_at_t))

            sample_step = space
            sampled_indices = list(range(0, len(avg_history), sample_step))
            sampled_values = [avg_history[i] for i in sampled_indices]

            all_histories.setdefault(algorithm, {})[pdsa] = sampled_values


plot_costs(all_histories,sampled_indices, p)



