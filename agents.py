import random
import numpy as np
# --------------------------------------------------- Message Class ----------------------------------------------------

# Message class: sender ID, receiver ID, and message content (value)
class Message:
    # Initialize a message with sender and receiver identifiers and the message value
    def __init__(self, sender_id, receiver_id, value, iteration,msg_type):
        self.sender_id = sender_id
        self.receiver_id = receiver_id
        self.value = value
        self.iteration = iteration
        self.type = msg_type

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
        self.iteration = 0
        self.current_costs = np.full(domain_size, np.inf)

    # Set a random initial value from the domain
    def set_initial_value(self, domain):
        return random.choice(list(domain))

    # Clear all messages in the agent's mailbox
    def clear_mailbox(self):
        self.mailbox.clear()

    def clear_last_mailbox(self):
        pass
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

    def send_messages(self, argument=None, msg_type="value"):
        if argument is None:
            argument = self.value
        for neighbor in self.neighbors:
            message = Message(self.id, neighbor.id, argument, self.iteration, msg_type)
            neighbor.mailbox.append(message)

    def compute_costs_from_last_it(self):
        costs = []
        for value in self.domain:
            cost = 0
            for message in self.mailbox:
                if (message.iteration == self.iteration - 1) and (message.type=="value") :
                    neighbor_id = message.sender_id
                    neighbor_value = message.value
                    matrix = self.cost_matrices[neighbor_id]
                    i = self.domain.index(value)
                    j = self.domain.index(neighbor_value)
                    cost += matrix[i][j]
            costs.append(cost)
        self.current_costs = costs

    # Decide whether to update assignment based on best local improvement and probability p
    def get_best_value(self,prob=1):
        best_value = self.value
        min_cost = min(self.current_costs)

        if min_cost <= self.current_costs[self.value]:
            best_values = [
                i for i, c in enumerate(self.current_costs)
                if c == min_cost
            ]
            best_values_minus_current = [v for v in best_values if v != self.value]
            if prob == 1:
                if best_values_minus_current:
                    best_value = random.choice(best_values_minus_current)
            else:
                if random.random() < prob:
                    if best_values_minus_current:
                        best_value =  random.choice(best_values_minus_current)

        return best_value

    # Determine if this agent has the highest score (gain or LR) among neighbors
    def has_highest_score(self, my_score, scores_received):
        for other_id, score in scores_received:
            # Tie-break by agent ID
            if score > my_score or (score == my_score and other_id < self.id):
                return False
        return True

    # algorithmic step to be implemented by subclasses
    def perform_phase1(self):
        pass

    def perform_phase2(self):
        pass


# ---------------------------------------------------- DSA Agent -------------------------------------------------------

# Agent for the DSA algorithm: probabilistically updates its assignment to reduce local cost
class DSAAgent(Agent):
    # Initialize DSA agent with probability p for accepting a new lower-cost value
    def __init__(self, agent_id, domainsize, p_dsa=0.7):
        super().__init__(agent_id, domainsize)
        self.p_dsa = p_dsa

    # Decide at every phase
    def perform_phase1(self):
        value = self.get_best_value(self.p_dsa)
        self.value = value

# Agent for the MGM algorithm: two-phase process to propose and apply the best local gain
class MGMAgent(Agent):
    def __init__(self, agent_id, domain):
        super().__init__(agent_id, domain)
        self.best_gain = 0  # Best gain from changing value
        self.best_value = self.value  # Value that yields best gain
        self.phase1_messages = []  # Messages to send in phase 1
        self.reduction = 0

    def decide_to_change(self):
        maximal = True
        for message in self.mailbox:
            if (message.iteration == self.iteration) and (message.type=="reduction"):
                if self.reduction < message.value:
                    maximal = False
                elif self.reduction == message.value:
                    if message.sender_id < self.id:
                        maximal = False
        return maximal

    def perform_phase1(self):
        best_alternative_value = self.get_best_value(1)
        self.reduction =  self.current_costs[self.value] - self.current_costs[best_alternative_value]

    def perform_phase2(self):
        self.send_messages(argument=self.reduction, msg_type="reduction")
        if self.decide_to_change():
            best_alternative_value = self.get_best_value(1)
            self.value = best_alternative_value

############ OBSOLETE ###############################
    # Phase 1: compute best gain and broadcast it to neighbors
    def prepare_phase1(self):
        current_cost = self.compute_cost(self.value)
        best_value = self.value
        best_cost = current_cost
        for v in self.domain:
            if v == self.value:
                continue
            c = self.compute_cost(v)
            if c < best_cost:
                best_cost = c
                best_value = v
        self.best_gain = current_cost - best_cost
        self.best_value = best_value
        # Create messages containing gain information
        return [Message(self.id, neighbor.id, self.best_gain) for neighbor in self.neighbors]

    # Phase 2: apply the assignment if this agent has the highest gain
    def apply_phase2(self):
        scores = [(msg.sender_id, msg.value) for msg in self.mailbox]
        if self.has_highest_score(self.best_gain, scores) and self.best_gain > 0:
            self.value = self.best_value

    # Execute appropriate phase based on phase number (even=prepare, odd=apply)
    def perform_phase(self, phase):
        if phase % 2 == 0:
            self.phase1_messages = self.prepare_phase1()
        else:
            self.apply_phase2()







