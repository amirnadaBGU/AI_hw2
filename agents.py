import random
import numpy as np
from jsonschema.exceptions import best_match
from networkx.classes import neighbors


# Message class
class Message:
    def __init__(self, sender_id, receiver_id, value, iteration,msg_type):
        self.sender_id = sender_id
        self.receiver_id = receiver_id
        self.value = value
        self.iteration = iteration
        self.type = msg_type
        self.read = False

# Base class for agents
class Agent():

    def __init__(self, agent_id, domain_size):
        self.id = agent_id
        self.domain = range(domain_size)
        self.value = self.set_initial_value(self.domain)
        self.neighbors = []  # List of neighboring Agent instances
        self.mailbox = []  # Received messages buffer
        self.cost_matrices = {}  # Cost matrices keyed by neighbor ID
        self.iteration = 0
        self.current_costs = np.full(domain_size, np.inf)

    def clear_read_messages(self):
        self.mailbox = [msg for msg in self.mailbox if not msg.read]

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
                    message.read = True
                    neighbor_id = message.sender_id
                    neighbor_value = message.value
                    matrix = self.cost_matrices[neighbor_id]
                    i = self.domain.index(value)
                    j = self.domain.index(neighbor_value)
                    cost += matrix[i][j]
            costs.append(cost)
        self.current_costs = costs

    def compute_costs_from_last_it_mgm2(self):
        costs = []
        for value in self.domain:
            cost = 0
            for message in self.mailbox:
                if (message.iteration == self.iteration) and (message.type=="value") :
                    message.read = True
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

        if min_cost <= self.current_costs[self.value] and self.current_costs[self.value]>0:
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

    # Determine if this agent has the highest score
    def has_highest_score(self, my_score, scores_received):
        for other_id, score in scores_received:
            # Tie-break by agent ID
            if score > my_score or (score == my_score and other_id < self.id):
                return False
        return True

# Agent for the DSA algorithm
class DSAAgent(Agent):
    # Initialize DSA agent with probability p for accepting a new lower-cost value
    def __init__(self, agent_id, domainsize, p_dsa=0.7):
        super().__init__(agent_id, domainsize)
        self.p_dsa = p_dsa

    # Decide at every phase
    def perform_phase1(self):
        value = self.get_best_value(self.p_dsa)
        self.value = value

# Agent for the MGM algorithm
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
            message.read = True
            if (message.iteration == self.iteration-1) and (message.type=="reduction"):
                if self.reduction < message.value:
                    maximal = False
                elif self.reduction == message.value:
                    if message.sender_id < self.id:
                        maximal = False
        return maximal

    def perform_phase1(self):
        best_alternative_value = self.get_best_value()
        self.reduction =  self.current_costs[self.value] - self.current_costs[best_alternative_value]
        self.send_messages(argument=self.reduction, msg_type="reduction")

    def perform_phase2(self):
        if self.decide_to_change():
            best_alternative_value = self.get_best_value(1)
            self.value = best_alternative_value
        self.send_messages(argument=self.value, msg_type="value")

# Agent for the MGM-2 algorithm
class MGM2Agent(MGMAgent):
    def __init__(self, agent_id, domain):
        super().__init__(agent_id, domain)
        self.potential_partner = None
        self.partner = None  # Partner agent in pair proposal
        self.proposals_received = []  # List of agents that proposed pairing
        self.best_pair_assignment = None  # Best assignment tuple (self_value, partner_value)
        self.reduction = 0  # Local reward value
        self.proposal_sent = False
        self.confirmed = False  # Confirmation flag after LR comparison
        self.has_maximal_reduction = False

    def clear_attributes_after_cycle(self):
        self.potential_partner = None
        self.partner = None
        self.proposals_received = []
        self.best_pair_assignment = None
        self.reduction = 0
        self.proposal_sent = False
        self.confirmed = False
        self.has_maximal_reduction = False
        self.changed = False

    def send_message_to_specific_agent(self, receiver, argument=None, msg_type="proposal"):
        if argument is None:
            argument = self.value
        message = Message(self.id, receiver.id, argument, self.iteration, msg_type)
        receiver.mailbox.append(message)

    def get_last_proposals(self):
        proposal_recieved = []
        for message in self.mailbox:
            if (message.iteration == self.iteration-1) and (message.type=="proposal"):
                message.read = True
                proposal_recieved.append(message)
        return proposal_recieved

    def get_partner_object(self,desired_proposal):
        return next((neighbor for neighbor in self.neighbors if neighbor.id == desired_proposal.sender_id), None)

    def compute_cost(self, value,partner_val=None):
        cost = 0
        for k, neighbor in enumerate(self.neighbors):
            if self.partner is not None: # YES PARTNER
                if neighbor.id == self.partner.id: # TALK ABOUT PARTNER
                    neighbor_value = partner_val
                else:
                    neighbor_value = neighbor.value # NOT TALK ABOUT PARTNER
            else:
                neighbor_value = neighbor.value # NO PARTNER
            cost += self.cost_matrices[neighbor.id][self.domain.index(value)][self.domain.index(neighbor_value)]
        return cost

    def compute_best_pair_assignment(self, partner):
        best_cost = float('inf')
        best_assignment = (self.value, partner.value)
        for v1 in self.domain:
            for v2 in partner.domain:
                c1 = self.compute_cost(v1,v2)
                c2 = partner.compute_cost(v2,v1)
                total_cost = c1 + c2 - self.cost_matrices[partner.id][v1][v2]
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_assignment = {self.id:v1, partner.id:v2}
        reduction = (self.compute_cost(self.value,partner.value)
                     + partner.compute_cost(partner.value,self.value)
                     - self.cost_matrices[partner.id][self.domain.index(self.value)][self.domain.index(partner.value)]- best_cost)
        return best_assignment, reduction

    def update_reduction_and_best_pair_assignment_from_message(self):
        for message in self.mailbox:
            if (message.iteration == self.iteration-1) and (message.type=="p2mgm2") and (message.sender_id == self.partner.id):
                message.read = True
                self.best_pair_assignment = message.value[0]
                self.reduction = message.value[1]

    def decide_to_change_partner(self):
        maximal = True
        for message in self.mailbox:
            if (message.iteration == self.iteration-1) and (message.type=="reduction"):
                message.read = True
                if message.sender_id == self.partner.id: # Dont read partner messages
                    continue
                if self.reduction < message.value: # If any other agent has higher reduction - not maximal
                    maximal = False
                elif self.reduction == message.value: # Tie break by agent ID
                    if message.sender_id < self.id: #  If same reduction, but lower ID - not maximal
                        maximal = False
        return maximal

    def get_changing_confirmation_from_partner(self):
        for message in self.mailbox:
            if (message.iteration == self.iteration-1) and (message.type=="changing") and (message.sender_id == self.partner.id):
                message.read = True
                return message.value
        return False

    def perform_phase1(self):
        if random.random() < 0.5 and self.neighbors:
            self.potential_partner = random.choice(self.neighbors) #second kind
            self.send_message_to_specific_agent(receiver=self.potential_partner,argument='Empty', msg_type="proposal")
            self.proposal_sent = True

    def perform_phase2(self):
        if self.proposal_sent is False: #Not Sender
            proposals = self.get_last_proposals()
            if len(proposals)>0: # Receiver
                self.partner = self.get_partner_object(random.choice(proposals))
                self.best_pair_assignment, self.reduction = self.compute_best_pair_assignment(self.partner)
                self.send_message_to_specific_agent(receiver=self.partner,
                                                    argument=[self.best_pair_assignment, self.reduction],
                                                    msg_type="p2mgm2")
                self.partner.partner = self

    def perform_phase3(self):
        if self.partner is not None and self.proposal_sent is True: # Sender
            self.update_reduction_and_best_pair_assignment_from_message()
        elif self.partner is None: # Single
            self.best_pair_assignment = self.get_best_value()
            self.reduction = self.current_costs[self.value] - self.current_costs[self.best_pair_assignment]
        self.send_messages(argument=self.reduction, msg_type="reduction")


    def perform_phase4(self):
        if self.partner is not None:
            self.has_maximal_reduction = self.decide_to_change_partner()
        else:
            self.has_maximal_reduction = self.decide_to_change()
        if self.partner is not None:
            self.send_message_to_specific_agent(receiver=self.partner,
                                                argument=self.has_maximal_reduction
                                                , msg_type="changing")

    def perform_phase5(self):
        if self.has_maximal_reduction:
            if self.partner is not None:
                if self.get_changing_confirmation_from_partner():
                    self.value = self.best_pair_assignment[self.id]
                    self.changed = True
            else:
                self.value = self.best_pair_assignment
                self.changed = True
        else:
            self.changed = False
        self.send_messages(argument=self.value, msg_type="value")







