from types import SimpleNamespace
import numpy as np
from scipy.optimize import minimize_scalar

# parameters
w = 1  # numeraire

# Define optimal firm behavior functions
def optimal_labor(w, p, A, gamma):
    return (p * A * gamma / w) ** (1 / (1 - gamma))

def optimal_output(l, A, gamma):
    return A * l ** gamma

def optimal_profit(w, p, A, gamma):
    l_star = optimal_labor(w, p, A, gamma)
    return (1 - gamma) / gamma * w * l_star

# Define optimal consumer behavior functions
def consumption_1(w, T, pi1, pi2, p1, p2, tau, alpha, l):
    income = w * l + T + pi1 + pi2
    return alpha * income / p1

def consumption_2(w, T, pi1, pi2, p1, p2, tau, alpha, l):
    income = w * l + T + pi1 + pi2
    return (1 - alpha) * income / (p2 + tau)

def utility(l, c1, c2, alpha, nu, epsilon):
    return np.log(c1 ** alpha * c2 ** (1 - alpha)) - nu * (l ** (1 + epsilon)) / (1 + epsilon)

# Compute optimal labor for the consumer
def optimal_labor_consumer(w, T, pi1, pi2, p1, p2, tau, alpha, nu, epsilon):
    def objective(l):
        c1 = consumption_1(w, T, pi1, pi2, p1, p2, tau, alpha, l)
        c2 = consumption_2(w, T, pi1, pi2, p1, p2, tau, alpha, l)
        return -(utility(l, c1, c2, alpha, nu, epsilon))
    result = minimize_scalar(objective, bounds=(0, 100), method='bounded')
    return result.x





#Question 3
def social_welfare_function(c1_star, c2_star, alpha, nu, l_star, epsilon, kappa):
    return utility(c1_star, c2_star, alpha, nu, l_star, epsilon) - kappa * c2_star

def objective(tau, w, p1, p2, alpha, nu, epsilon, kappa):
    A = 1.0
    gamma = 0.5
    pi1_star = optimal_profit(w, p1, A, gamma)
    pi2_star = optimal_profit(w, p2, A, gamma)
    
    # Calculate implied T based on tau and the optimal consumption
    l_opt = optimal_labor_consumer(w, 0, pi1_star, pi2_star, p1, p2, tau, alpha, nu, epsilon)
    c2_star = consumption_2(w, 0, pi1_star, pi2_star, p1, p2, tau, alpha, l_opt)
    T_opt = tau * c2_star
    
    # Recalculate labor and consumption with the updated T
    l_opt = optimal_labor_consumer(w, T_opt, pi1_star, pi2_star, p1, p2, tau, alpha, nu, epsilon)
    c1_star = consumption_1(w, T_opt, pi1_star, pi2_star, p1, p2, tau, alpha, l_opt)
    c2_star = consumption_2(w, T_opt, pi1_star, pi2_star, p1, p2, tau, alpha, l_opt)
    
    # Calculate social welfare function
    swf = social_welfare_function(c1_star, c2_star, alpha, nu, l_opt, epsilon, kappa)
    return -swf  # We minimize the negative of SWF to maximize SWF



