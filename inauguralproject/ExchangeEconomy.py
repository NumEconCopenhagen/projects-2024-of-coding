from types import SimpleNamespace

class ExchangeEconomyClass:

    def __init__(self):

        par = self.par = SimpleNamespace()

        # a. preferences
        par.alpha = 1/3
        par.beta = 2/3

        # b. endowments
        par.w1A = 0.8
        par.w2A = 0.3

    def utility_A(self,x1A,x2A, alpha):
        (x1A**alpha) * (x2A**(1-alpha))

    def utility_B(self,x1B,x2B, beta):
        (x1B**beta) * (x2B**(1-beta))

    def demand_A(self,p1, alpha, w1A, w2A):
        p2 = 1
        x1A = alpha * ((p1*w1A + p2*w2A)/p1)    
        x2A = (1 - alpha) * ((p1*w1A + p2*w2A)/p2)    

        return x1A, x2A
    
    def demand_B(self,p1, beta, w1A, w2A):
        w1B = 1 - w1A
        w2B = 1 - w1B
        p2 = 1

        x1B = beta * ((p1*w1B+p2*w2B)/p1)
        x2B = (1 - beta) * ((p1*w1B+p2*w2B)/p2)

        return x1B, x2B

    def check_market_clearing(self,p1):

        par = self.par

        x1A,x2A = self.demand_A(p1, par.alpha, par.w1A, par.w2A)
        x1B,x2B = self.demand_B(p1, par.beta, par.w1A, par.w2A)

        eps1 = x1A-par.w1A + x1B-(1-par.w1A)
        eps2 = x2A-par.w2A + x2B-(1-par.w2A)

        return eps1,eps2