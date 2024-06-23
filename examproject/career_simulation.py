# career_simulation.py
import numpy as np
import matplotlib.pyplot as plt

class CareerSimulation:
    def __init__(self, J=3, N=10, K=10000, sigma=2, v=None):
        self.J = J  # Number of careers
        self.N = N  # Number of graduates
        self.K = K  # Number of simulations
        self.sigma = sigma  # Noise standard deviation
        self.v = v if v is not None else np.array([1, 2, 3])  # Intrinsic utilities

        self.results = np.zeros((N, J))
        self.expected_utilities = np.zeros(N)
        self.realized_utilities = np.zeros(N)
    
    def simulate(self):
        for i in range(self.N):
            Fi = i + 1
            utilities = np.zeros((self.J, self.K))
            for j in range(self.J):
                noise = np.random.normal(0, self.sigma, (Fi, self.K))
                utilities[j] = self.v[j] + noise.mean(axis=0)

            chosen_careers = np.argmax(utilities, axis=0)
            for j in range(self.J):
                self.results[i, j] = np.mean(chosen_careers == j)

            # Calculating expected utility for each graduate
            self.expected_utilities[i] = np.mean(np.max(utilities, axis=0))

            # Calculating realized utility using the chosen career
            chosen_utility_values = self.v[chosen_careers] + np.random.normal(0, self.sigma, self.K)
            self.realized_utilities[i] = np.mean(chosen_utility_values)

    def plot_results(self):
        # Plotting career choices
        fig, ax = plt.subplots(figsize=(10, 8))
        for j in range(self.J):
            ax.plot(range(1, self.N + 1), self.results[:, j], label=f'Career {j + 1}', marker='o')
        ax.set_xlabel('Graduate (Number of Friends)')
        ax.set_ylabel('Share Choosing Career')
        ax.set_title('Share of Each Career Choice by Graduate')
        ax.legend()
        plt.show()

        # Plotting expected utilities
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, self.N + 1), self.expected_utilities, marker='o', color='blue')
        plt.title('Average Subjective Expected Utility by Graduate')
        plt.xlabel('Graduate (Number of Friends)')
        plt.ylabel('Expected Utility')
        plt.grid(True)
        plt.show()

        # Plotting realized utilities
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, self.N + 1), self.realized_utilities, marker='o', color='green')
        plt.title('Average Ex-Post Realized Utility by Graduate')
        plt.xlabel('Graduate (Number of Friends)')
        plt.ylabel('Realized Utility')
        plt.grid(True)
        plt.show()

