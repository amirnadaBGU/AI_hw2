import random

# --------------------------------------------------- Message Class ----------------------------------------------------

# Message class: sender ID, receiver ID, and message content (value)
class Message:
    # Initialize a message with sender and receiver identifiers and the message value
    def __init__(self, sender_id, receiver_id, value):
        self.sender_id = sender_id
        self.receiver_id = receiver_id
        self.value = value

# ---------------------------------------------------- Agent Class -----------------------------------------------------

# Abstract base class for agents
class Agent():
    # agent_id: unique identifier
    # domain: list of possible values the agent can hold
    # Default: pick a random starting value, set up neighbors, mailbox, and cost matrices
    def __init__(self, agent_id, domain_size):
        self.id = agent_id
        self.domain = range(domain_size)
        self.value = self.set_initial_value(self.domain)
        self.neighbors = []  # List of neighboring Agent instances
        self.mailbox = []  # Received messages buffer
        self.cost_matrices = {}  # Cost matrices keyed by neighbor ID

    # Set a random initial value from the domain
    def set_initial_value(self, domain):
        return random.choice(list(domain))

    # Clear all messages in the agent's mailbox
    def clear_mailbox(self):
        self.mailbox.clear()

    # Receive a message from a neighbor and append it to the mailbox
    def receive_message(self, message):
        self.mailbox.append(message)

    # Generate current value to all neighbors
    def generate_messages(self):
        return [Message(self.id, neighbor.id, self.value) for neighbor in self.neighbors]

    # Compute the cost for choosing a given value, based on messages received and stored cost matrices
    def compute_cost(self, value):
        cost = 0
        for message in self.mailbox:
            neighbor_id = message.sender_id
            neighbor_value = message.value
            matrix = self.cost_matrices[neighbor_id]
            i = self.domain.index(value)
            j = self.domain.index(neighbor_value)
            cost += matrix[i][j]
        return cost

    # Determine if this agent has the highest score (gain or LR) among neighbors
    def has_highest_score(self, my_score, scores_received):
        for other_id, score in scores_received:
            # Tie-break by agent ID
            if score > my_score or (score == my_score and other_id < self.id):
                return False
        return True


    # algorithmic step to be implemented by subclasses
    def perform_phase(self, phase):
        pass


# ---------------------------------------------------- DSA Agent -------------------------------------------------------

# Agent for the DSA algorithm: probabilistically updates its assignment to reduce local cost
class DSAAgent(Agent):
    # Initialize DSA agent with probability p for accepting a new lower-cost value
    def __init__(self, agent_id, domainsize, p=0.7):
        super().__init__(agent_id, domainsize)
        self.p = p

    # Decide whether to update assignment based on best local improvement and probability p
    def decide(self):
        current_cost = self.compute_cost(self.value)
        best_value = self.value
        best_cost = current_cost
        # Evaluate cost for each possible value
        for v in self.domain:
            if v == self.value:
                continue
            c = self.compute_cost(v)
            if c < best_cost:
                best_cost = c
                best_value = v
        # With probability p, switch to the best value if it's an improvement
        if best_value != self.value and random.random() < self.p:
            self.value = best_value

    # Decide at every phase
    def perform_phase(self, phase):
        self.decide()




