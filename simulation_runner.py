import random

from DCOP import DCOPInstance
from agents import Agent, DSAAgent
from simulation import Simulation, plot_costs

if __name__ == '__main__':
    p1 = 0.75
    DCOP = DCOPInstance(30, 5, p1, 42)
    Sim = Simulation(DCOP,'MGM2',p_dsa = 0.7)
    Sim.run(steps=125)
    history = {"DSA": {}}
    history['DSA'][1] = Sim.history
    plot_costs(history,p1)
