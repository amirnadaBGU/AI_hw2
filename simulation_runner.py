if __name__ == '__main__':
    from DCOP import DCOPInstance
    from agents import Agent, DSAAgent
    from simulation import Simulation

    DCOP = DCOPInstance(30, 5, 0.25, 42)
    Sim = Simulation(DCOP,1)
    Sim.run(steps=125)
