from agents import DSAAgent, MGMAgent, MGM2Agent
import copy
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# ------------------------------------------------- Simulation Class ---------------------------------------------------

class Simulation:
    def __init__(self, DCOP,agent_type,p_dsa=None):
        self.DCOP = DCOP
        self.agent_type = agent_type
        self.agents = self.build_agents_from_problem(DCOP,p_dsa)
        self.iteration = 0
        self.history = []


    def build_agents_from_problem(self, DCOP, p_dsa):
        agents = []
        for i in range(DCOP.num_agents):
            if self.agent_type =='DSA':
                agent = DSAAgent(i, DCOP.domain_size, p_dsa=p_dsa)
            elif self.agent_type  =='MGM':
                agent = MGMAgent(i, DCOP.domain_size)
            elif self.agent_type  =='MGM2':
                agent = MGM2Agent(i, DCOP.domain_size)
            else:
                raise ValueError("Unknown algorithm type")

            agents.append(agent)

        # Attach neighbors and their cost matrices
        for i, agent in enumerate(agents):
            for j in DCOP.neighbors_map[i]:
                agent.neighbors.append(agents[j])
                agent.cost_matrices[agents[j].id] = copy.deepcopy(DCOP.cost_matrices[i][j])
        return agents

    # Run the simulation
    def run(self,steps):
        # Generate new messages
        if self.agents[0].__class__ in [DSAAgent, MGMAgent, MGM2Agent]:
            for agent in self.agents:
                agent.send_messages()

        while self.iteration < steps:
            self.iteration += 1

            self.global_cost = self.compute_global_cost()
            self.history.append(self.global_cost)

            if self.agents[0].__class__ in [DSAAgent]:
                for agent in self.agents:
                    agent.iteration = self.iteration
                    agent.compute_costs_from_last_it()
                    agent.perform_phase1()
                    agent.clear_read_messages()
                    agent.send_messages()

            elif self.agents[0].__class__ in [MGMAgent]:
                if self.iteration % 2 == 1:
                    for agent in self.agents:
                        agent.iteration = self.iteration
                        agent.compute_costs_from_last_it()
                        agent.perform_phase1()
                        agent.clear_read_messages()
                        agent.send_messages()
                else:
                    for agent in self.agents:
                        agent.iteration = self.iteration
                        agent.perform_phase2()

            elif self.agents[0].__class__ in [MGM2Agent]:
                    for agent in self.agents:
                        if agent.iteration % 5 == 0:
                            agent.compute_costs_from_last_it()
                            agent.clear_read_messages()
                            agent.perform_phase1()
                            agent.iteration = self.iteration
                        elif agent.iteration % 5 == 1:
                            agent.perform_phase2()
                            agent.iteration = self.iteration
                        elif agent.iteration % 5 == 2:
                            agent.perform_phase3()
                            agent.iteration = self.iteration
                        elif agent.iteration % 5 == 3:
                            agent.perform_phase4()
                            agent.iteration = self.iteration
                        elif agent.iteration % 5 == 4:
                            agent.perform_phase5()
                            agent.iteration = self.iteration
                            agent.clear_attributes_after_cycle()

    def compute_global_cost(self):
        total = 0
        counted = set()
        for agent in self.agents:
            for neighbor in agent.neighbors:
                key = tuple(sorted((agent.id, neighbor.id)))
                if key in counted:
                    continue
                matrix = agent.cost_matrices[neighbor.id]
                i = agent.domain.index(agent.value)
                j = neighbor.domain.index(neighbor.value)
                total += matrix[i][j]
                counted.add(key)
        return total

# Plot global cost histories for different algorithms
def plot_costs(all_histories,indices, k):
    plt.figure()
    all_values = []

    # Plot DSA histories (record every iteration)
    dsa_histories = all_histories.get("DSA", {})
    for p, history in dsa_histories.items():
        plt.plot(indices, history, label=f"DSA p={p}", linewidth=0.6)
        all_values.extend(history)

    # Plot MGM (step every 2 iterations)
    mgm_history = all_histories.get("MGM", {}).get(None)
    if mgm_history:
        plt.step(indices, mgm_history, where="post", label="MGM", linewidth=0.6)
        all_values.extend(mgm_history)

    # Plot MGM2 (step every 5 iterations)
    mgm2_history = all_histories.get("MGM2", {}).get(None)
    if mgm2_history:
        plt.step(indices, mgm2_history, where="post", label="MGM2", linewidth=0.6)
        all_values.extend(mgm2_history)

    plt.xlabel("Iteration")
    plt.ylabel("Global Cost")
    plt.title(f"DSA vs MGM | p1={k}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filename = f"DSA_vs_MGM_p1{k}.png"
    plt.savefig(filename, dpi=300)
    plt.show()