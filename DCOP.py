import numpy as np
import random
import copy


# ------------------------------------------------ Constraint Builder --------------------------------------------------

# Build cost matrix between two agents, either random or graph-coloring
def create_constraint_matrix(domain_size,low_val=1, high_val=10,p2=1):
    matrix = [[0] * domain_size for _ in range(domain_size)]
    for i in range(domain_size):
        for j in range(domain_size):
                if random.random() <= p2:
                    matrix[i][j] = random.randint(low_val, high_val)
    return np.array(matrix)


# --------------------------------------------- Problem Instance Class--------------------------------------------------

class DCOPInstance:
    # Initialize instances with given parameters
    def __init__(self, num_agents, domain_size, p1,p2, seed):
        random.seed(seed)
        np.random.seed(seed)
        self.num_agents = num_agents
        self.domain_size = domain_size
        self.domain = list(range(domain_size))
        self.neighbors_map = {i: [] for i in range(self.num_agents)}  # Adjacency list
        self.cost_matrices = {i: {} for i in range(self.num_agents)}  # Pairwise cost matrices

        # Construct random constrains with probability p1 and assign cost matrices
        # Cost matrix from j to i is the transpose of cost matrix from i to j.
        # Assume each variable see itself as the rows of the matrix.
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                if random.random() < p1:
                    self.neighbors_map[i].append(j)
                    self.neighbors_map[j].append(i)
                    matrix = create_constraint_matrix(domain_size=domain_size,p2=p2)
                    self.cost_matrices[i][j] = np.array(matrix)
                    self.cost_matrices[j][i] = np.array(np.transpose(matrix)) #TODO check if transpose is correct



if __name__ == "__main__":
    print(create_constraint_matrix(5))
    DCOP = DCOPInstance(3, 2, 0.7, 42)

