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
            if (message.iteration == self.iteration-1) and (message.type=="reduction"):
                if self.reduction < message.value:
                    maximal = False
                elif self.reduction == message.value:
                    if message.sender_id < self.id:
                        maximal = False
        return maximal

    def perform_phase1(self):
        best_alternative_value = self.get_best_value(1)
        self.reduction =  self.current_costs[self.value] - self.current_costs[best_alternative_value]
        self.send_messages(argument=self.reduction, msg_type="reduction")

    def perform_phase2(self):
        if self.decide_to_change():
            best_alternative_value = self.get_best_value(1)
            self.value = best_alternative_value
        self.send_messages(argument=self.value, msg_type="value")


# Agent for the MGM-2 algorithm: extends MGM with pair assignments, 5 phases
class MGM2Agent(MGMAgent):
    def __init__(self, agent_id, domain):
        super().__init__(agent_id, domain)
        self.partner = None  # Partner agent in pair proposal
        self.proposals_received = []  # List of agents that proposed pairing
        self.lrs_received = []  # List of LRs received
        self.best_pair_assignment = None  # Best assignment tuple (self_value, partner_value)
        self.lr_value = 0  # Local reward value
        self.confirmed = False  # Confirmation flag after LR comparison
        self.phase1_messages = []

    # Execute assignment for the agreed pair
    def perform_pair_assignment(self):
        if self.best_pair_assignment and self.partner:
            self.value = self.best_pair_assignment[0]
            self.partner.value = self.best_pair_assignment[1]

    def perform_phase(self, phase):
        phase_mod = phase % 5

        # Phase 0: send pairing proposal to a random neighbor with probability 0.5
        if phase_mod == 0:
            if random.random() < 0.5 and self.neighbors:
                self.partner = random.choice(self.neighbors)
                self.phase1_messages = [Message(self.id, self.partner.id, "proposal")]
            else:
                self.partner = None
                self.proposals_received = []

        # Phase 1: receive proposals and compute pair assignment or compute local LR if no pair
        elif phase_mod == 1:
            if self.proposals_received:
                self.partner = random.choice(self.proposals_received)
                self.best_pair_assignment, self.lr_value = self.compute_best_pair_assignment(self.partner)
                self.phase1_messages = [
                    Message(self.id, self.partner.id, ("pair_assignment", self.best_pair_assignment, self.lr_value)),
                    Message(self.id, self.partner.id, "ack")
                ]
            elif self.partner is not None:
                # Proposed but no acknowledgment: fallback to local LR
                self.lr_value = self.compute_local_lr()
                self.best_pair_assignment = None
                self.phase1_messages = []
            else:
                # No proposals: compute local LR normally
                self.lr_value = self.compute_local_lr()
                self.best_pair_assignment = None
                self.phase1_messages = []

        # Phase 2: broadcast LR to all neighbors
        elif phase_mod == 2:
            self.lrs_received = []
            self.phase1_messages = [
                Message(self.id, neighbor.id, ("lr", self.lr_value)) for neighbor in self.neighbors
            ]

        # Phase 3: confirm if this agent has the highest LR among received
        elif phase_mod == 3:
            self.confirmed = self.has_highest_score(self.lr_value, self.lrs_received)

        # Phase 4: finalize assignment: either pair assignment or local best
        elif phase_mod == 4:
            if (
                    self.confirmed and
                    self.partner and
                    self.partner.confirmed and
                    self.partner.partner == self
            ):
                self.perform_pair_assignment()
            elif self.confirmed and (self.partner is None or not self.partner.confirmed):
                # Solo improvement if confirmed but no valid partner
                best_value = self.value
                best_cost = self.compute_cost(self.value)
                for v in self.domain:
                    c = self.compute_cost(v)
                    if c < best_cost:
                        best_cost = c
                        best_value = v
                if best_value != self.value:
                    self.value = best_value

            # Reset negotiation state after each 5-phase cycle
            self.partner = None
            self.proposals_received = []
            self.lrs_received = []
            self.best_pair_assignment = None
            self.confirmed = False
            self.phase1_messages = []

    # Override receive_message to handle complex message types for MGM-2
    def receive_message(self, message):
        if message.value == "proposal":
            proposer_obj = next((agent for agent in self.neighbors if agent.id == message.sender_id), None)
            if proposer_obj:
                self.proposals_received.append(proposer_obj)
        elif isinstance(message.value, tuple):
            tag = message.value[0]
            if tag == "pair_assignment":
                _, assignment, lr_value = message.value
                self.best_pair_assignment = assignment
                self.lr_value = lr_value
                sender_obj = next((agent for agent in self.neighbors if agent.id == message.sender_id), None)
                if sender_obj:
                    self.partner = sender_obj
            elif tag == "lr":
                _, lr = message.value
                self.lrs_received.append((message.sender_id, lr))
        elif message.value == "ack":
            sender_obj = next((agent for agent in self.neighbors if agent.id == message.sender_id), None)
            if sender_obj:
                self.partner = sender_obj
        else:
            self.mailbox.append(message)

    # Compute best joint assignment for this agent and partner
    def compute_best_pair_assignment(self, partner):
        best_cost = float('inf')
        best_assignment = (self.value, partner.value)
        for v1 in self.domain:
            for v2 in partner.domain:
                c1 = self.compute_cost(v1)
                c2 = partner.compute_cost(v2)
                total_cost = c1 + c2
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_assignment = (v1, v2)
        lr = self.compute_cost(self.value) + partner.compute_cost(partner.value) - best_cost
        return best_assignment, lr

    # Compute LR as the cost reduction achievable by solo change
    def compute_local_lr(self):
        best_cost = self.compute_cost(self.value)
        for v in self.domain:
            c = self.compute_cost(v)
            if c < best_cost:
                best_cost = c
        return self.compute_cost(self.value) - best_cost







