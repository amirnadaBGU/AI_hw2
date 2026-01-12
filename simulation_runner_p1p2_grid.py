import random
import numpy as np
from DCOP import DCOPInstance
from agents import Agent, DSAAgent
from simulation import Simulation, plot_costs
import matplotlib.pyplot as plt
if __name__ == '__main__':
    p1 = [0.2,0.5]
    p2 = np.linspace(0.1, 1, 10)
    algorithms = [
        ("DSA", 0.7),
        ("MGM", None),
        ("MGM2", None),
    ]

    for p_1 in p1:
        all_costs = {}
        for p_2 in p2:
            print("p2:",p_2)
            all_costs[p_2] = {}
            problem_instances = [DCOPInstance(30, 10, p_1, p_2, seed=random.randint(1,100000)) for run in range(50)]
            for alg_name, pdsa in algorithms:
                print("alg:", alg_name)
                costs = []
                pr_number = 1
                for dcop in problem_instances:
                    print("problem:", pr_number)
                    pr_number += 1
                    Sim = Simulation(dcop, alg_name, p_dsa=pdsa)
                    Sim.run(steps=125)
                    costs.append(Sim.global_cost)

                total_cost = sum(costs)/ len(costs)
                all_costs[p_2][alg_name] = total_cost


        def plot_all_costs(all_costs_dict):
            plt.figure()  # יצירת חלון חדש לכל p1
            x_vals = sorted(all_costs_dict.keys())

            if not x_vals:
                print("No data to plot")
                return

            first_key = x_vals[0]
            algs = list(all_costs_dict[first_key].keys())

            for alg in algs:
                y_vals = [all_costs_dict[x][alg] for x in x_vals]
                plt.plot(x_vals, y_vals, marker='o', label=alg)

            plt.xlabel('p2')
            plt.ylabel('Total Cost')
            plt.title(f'Costs vs p2 (p1={p_1})')
            plt.legend()
            plt.grid(True)
            plt.show()


        plot_all_costs(all_costs)





