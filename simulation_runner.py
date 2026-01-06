from DCOP import DCOPInstance
from agents import Agent, DSAAgent
from simulation import Simulation, plot_costs

if __name__ == '__main__':
    k = 0.25
    DCOP = DCOPInstance(30, 5, k, 42)
    Sim = Simulation(DCOP,0.7)
    Sim.run(steps=125)
    history = {"DSA": {}}
    history['DSA'][1] = Sim.history
    plot_costs(history,k)
