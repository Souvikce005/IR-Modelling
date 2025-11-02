import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#dr = k(theta - r)dt + sigma*dw

# r(t+1) = r(t) + k(theta - r(t))*delta_t + sigma*sqrt(delta_t)*e

def random_generator(): 
    e = np.random.normal(0,1)
    return e

def simulate_paths(r0, k, theta, sigma, delta_t, sim_period):    
    rates = []
    rates.append(r0)
    for i in range(sim_period): 
        stoch_term = random_generator()
        next_rate = rates[i] + k*(theta - rates[i])*delta_t + sigma*np.sqrt(delta_t)*stoch_term
        print(next_rate)
        rates.append(next_rate)
    return rates

def main():
    r0 = 0.035
    k = 0.15
    theta = 0.04
    sigma = 0.01
    delta_t = 1/252
    sim_period = 21
    rates = simulate_paths(r0, k, theta, sigma, delta_t, sim_period)
    print(rates)
    plt.plot(rates)
    plt.title('Simulated Interest Rate Paths')
    plt.xlabel('Time Steps')
    plt.ylabel('Interest Rate')
    plt.show()

if __name__ == "__main__":
    main()
