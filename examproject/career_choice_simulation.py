import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace

class CareerChoiceSimulation:
    def __init__(self, num_careers=3, num_graduates=10, num_simulations=10000, sigma=2, intrinsic_utilities=np.array([1, 2, 3]), switching_cost=1):
        # Initialize simulation parameters
        self.par = SimpleNamespace()
        self.par.J = num_careers
        self.par.N = num_graduates
        self.par.K = num_simulations
        self.par.sigma = sigma
        self.par.v = intrinsic_utilities
        self.par.c = switching_cost

        # Initialize arrays for the simulation
        self.choices = np.zeros((self.par.N, self.par.J, self.par.K))
        self.subjective_utilities = np.zeros((self.par.N, self.par.K))
        self.realized_utilities = np.zeros((self.par.N, self.par.K))

    def simulate_first_year(self):
        for i in range(self.par.N):
            F_i = i + 1
            for k in range(self.par.K):
                epsilon_friends = np.random.normal(0, self.par.sigma, (F_i, self.par.J))
                epsilon_individual = np.random.normal(0, self.par.sigma, self.par.J)

                friend_utilities = self.par.v + epsilon_friends.mean(axis=0)
                total_expected_utility = friend_utilities + epsilon_individual

                choice = np.argmax(total_expected_utility)
                self.choices[i, choice, k] = 1
                self.subjective_utilities[i, k] = friend_utilities[choice]
                self.realized_utilities[i, k] = self.par.v[choice] + epsilon_individual[choice]

    def simulate_second_year(self):
        self.second_year_choices = np.zeros((self.par.N, self.par.J, self.par.K))
        self.second_year_subjective_utilities = np.zeros((self.par.N, self.par.K))
        self.second_year_realized_utilities = np.zeros((self.par.N, self.par.K))
        self.switch_counts = np.zeros((self.par.N, self.par.K))

        for i in range(self.par.N):
            for k in range(self.par.K):
                first_year_choice = np.argmax(self.choices[i, :, k])
                epsilon_friends_second = np.random.normal(0, self.par.sigma, (i + 1, self.par.J))
                epsilon_individual_second = np.random.normal(0, self.par.sigma, self.par.J)

                friend_utilities_second = self.par.v + epsilon_friends_second.mean(axis=0)
                new_priors = np.where(np.arange(self.par.J) != first_year_choice, friend_utilities_second - self.par.c, self.realized_utilities[i, k])
                total_expected_utility_second = new_priors + epsilon_individual_second

                second_choice = np.argmax(total_expected_utility_second)
                self.second_year_choices[i, second_choice, k] = 1
                if second_choice != first_year_choice:
                    self.switch_counts[i, k] = 1

                self.second_year_subjective_utilities[i, k] = new_priors[second_choice]
                self.second_year_realized_utilities[i, k] = self.par.v[second_choice] + epsilon_individual_second[second_choice]

    def calculate_averages(self):
        # Calculate averages for the first and second years
        self.average_choices = self.choices.mean(axis=2)
        self.average_subjective_utilities = self.subjective_utilities.mean(axis=1)
        self.average_realized_utilities = self.realized_utilities.mean(axis=1)

        self.average_second_year_choices = self.second_year_choices.mean(axis=2)
        self.average_second_year_subjective_utilities = self.second_year_subjective_utilities.mean(axis=1)
        self.average_second_year_realized_utilities = self.second_year_realized_utilities.mean(axis=1)
        self.average_switches = self.switch_counts.mean(axis=1)

    def plot_results(self):
        # Plot various metrics
        fig, ax = plt.subplots(4, 1, figsize=(10, 20))

        # Plot for each career in the second year
        for j in range(self.average_second_year_choices.shape[1]):
            ax[0].plot(range(1, self.average_second_year_choices.shape[0] + 1), self.average_second_year_choices[:, j], label=f'Career {j+1}', marker='o')
        ax[0].set_xlabel('Graduate (Number of Friends)')
        ax[0].set_ylabel('Share Choosing Career')
        ax[0].set_title('Average Second Year Choice Share per Career')
        ax[0].legend()
        ax[0].grid(True)

        # Plot for average second year subjective utility
        ax[1].plot(range(1, self.average_second_year_subjective_utilities.shape[0] + 1), self.average_second_year_subjective_utilities, marker='o', color='blue')
        ax[1].set_title('Average Second Year Subjective Utility')
        ax[1].set_xlabel('Graduate (Number of Friends)')
        ax[1].set_ylabel('Subjective Utility')
        ax[1].grid(True)

        # Plot for average second year realized utility
        ax[2].plot(range(1, self.average_second_year_realized_utilities.shape[0] + 1), self.average_second_year_realized_utilities, marker='o', color='green')
        ax[2].set_title('Average Second Year Realized Utility')
        ax[2].set_xlabel('Graduate (Number of Friends)')
        ax[2].set_ylabel('Realized Utility')
        ax[2].grid(True)

        # Plot for average switch share
        ax[3].plot(range(1, self.average_switches.shape[0] + 1), self.average_switches, marker='o', color='red')
        ax[3].set_title('Average Switch Share')
        ax[3].set_xlabel('Graduate (Number of Friends)')
        ax[3].set_ylabel('Switch Share')
        ax[3].grid(True)

        plt.tight_layout()
        plt.show()
