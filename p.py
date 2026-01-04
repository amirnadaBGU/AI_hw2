from agents import DSAAgent, MGMAgent, MGM2Agent
import numpy as np
import random
import copy


# ------------------------------------------------ Constraint Builder --------------------------------------------------

# Build cost matrix between two agents, either random or graph-coloring
def create_constraint_matrix(domain, coloring=False):
    size = len(domain)
    matrix = [[0] * size for _ in range(size)]
    for i in range(size):
        for j in range(i, size):
            if coloring:
                # Zero cost for different colors, high self-penalty to avoid same-color assignment
                val = random.randint(100, 200) if i == j else 0
            else:
                # Fully random costs
                val = random.randint(100, 200)
            matrix[i][j] = matrix[j][i] = val
    return matrix


# --------------------------------------------- Problem Instance Class--------------------------------------------------

class ProblemInstance:
    # Initialize instances with given parameters
    def __init__(self, num_agents, domain, k, coloring, seed):
        random.seed(seed)
        np.random.seed(seed)
        self.initial_values = []  # Random starting assignments
        self.neighbors_map = {i: [] for i in range(num_agents)}  # Adjacency list
        self.cost_matrices = {i: {} for i in range(num_agents)}  # Pairwise cost matrices

        # Generate random initial values for each agent
        for i in range(num_agents):
            self.initial_values.append(random.choice(domain))

        # Construct random graph edges with probability k and assign cost matrices
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                if random.random() < k:
                    self.neighbors_map[i].append(j)
                    self.neighbors_map[j].append(i)
                    matrix = create_constraint_matrix(domain, coloring=coloring)
                    self.cost_matrices[i][j] = matrix
                    self.cost_matrices[j][i] = matrix


def build_agents_from_problem(problem, domain, algorithm, p=None):
    # Create agents of specified algorithm type and link them according to problem
    agents = []
    for i, init_value in enumerate(problem.initial_values):
        if algorithm == "DSA":
            agent = DSAAgent(i, domain, p=p)
        elif algorithm == "MGM":
            agent = MGMAgent(i, domain)
        elif algorithm == "MGM2":
            agent = MGM2Agent(i, domain)
        else:
            raise ValueError("Unknown algorithm type")
        agent.value = init_value
        agents.append(agent)

    # Attach neighbors and their cost matrices
    for i, agent in enumerate(agents):
        for j in problem.neighbors_map[i]:
            agent.neighbors.append(agents[j])
            agent.cost_matrices[j] = copy.deepcopy(problem.cost_matrices[i][j])
    return agents

if __name__ == "__main__":
    print(create_constraint_matrix(5))
    DCOP = ProblemInstance(2, [0,1], 1, 42)
    print('s')
