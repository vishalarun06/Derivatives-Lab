import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.optimize import newton
from scipy.optimize import brentq
import pandas as pd
from datetime import date

class Option:
    def __init__(self, S, K, T, r, sigma, option_type="call", exercise_type="european"):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma

        if option_type.lower() not in ["call","put"]:
            raise ValueError("Option type must be 'call' or 'put'!")
        self.option_type = option_type.lower()

        if exercise_type.lower() not in ["european","american"]:
            raise ValueError("Exercise type must be 'american' or 'european'!")
        self.exercise_type = exercise_type.lower()

    def price_black_scholes(self):
        if self.exercise_type == "american":
            raise ValueError("Black-Scholes formula is not valid for American options. Use a binomial tree instead.")
        d1 = (np.log(self.S/self.K)+self.T*(self.r+(self.sigma**2)*0.5))/(self.sigma*np.sqrt(self.T))
        d2 = d1 - self.sigma*np.sqrt(self.T)
    
        if self.option_type == "call":
            call = self.S*norm.cdf(d1) - self.K*np.exp(-self.r*self.T)*norm.cdf(d2)
            return call

        elif self.option_type == "put":
            put = self.K*np.exp(-self.r*self.T)*norm.cdf(-d2) - self.S*norm.cdf(-d1)
            return put

    def get_greeks(self):
        if self.exercise_type == "american":
            raise ValueError("Black-Scholes formula is not valid for American options. Use a binomial tree instead.")
        d1 = (np.log(self.S/self.K)+self.T*(self.r+0.5*(self.sigma**2)))/(self.sigma*np.sqrt(self.T))
        d2 = d1 - self.sigma*np.sqrt(self.T)

        
        gamma = norm.pdf(d1)/(self.S*self.sigma*np.sqrt(self.T))
        vega = self.S * norm.pdf(d1) * np.sqrt(self.T)
        if self.option_type == "call":
            delta = norm.cdf(d1)
            theta = - (self.S * norm.pdf(d1) * self.sigma) / (2*np.sqrt(self.T)) - self.r * self.K * np.exp(-self.r*self.T) * norm.cdf(d2)
            rho = self.K * self.T * np.exp(-self.r*self.T)*norm.cdf(d2)
        else: # For a put
            delta = norm.cdf(d1) - 1
            theta = - (self.S * norm.pdf(d1) * self.sigma) / (2*np.sqrt(self.T)) + self.r * self.K * np.exp(-self.r*self.T) * norm.cdf(-d2)
            rho = -self.K * self.T * np.exp(-self.r*self.T)*norm.cdf(-d2)
        
        return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}
    
    def get_implied_volatility(self, market_price, max_vol=20):
        if self.option_type == "call":
            intrinsic_value = max(0, self.S - self.K * np.exp(-self.r * self.T))
        else:
            intrinsic_value = max(0, self.K * np.exp(-self.r * self.T) - self.S)
        
        if market_price < intrinsic_value:
            return np.nan
        def func(sigma):
            return (Option(self.S, self.K, self.T, self.r, sigma).price_black_scholes() - market_price)
        #def fprime(sigma):
            #return get_greeks(self, sigma)["vega"]
        #initial = np.sqrt(2*np.pi/self.T) * market_price/self.S
        a=1e-6
        b=1
        while func(b) < 0:
            b = 2*b
            if b > max_vol:
                return np.nan
        try:
            return brentq(func, a, b)
        except (ValueError, RuntimeError):
            return np.nan

    def plot_option_price_vs_stock_price(self):
        S_values = range(int(round(0.5*self.S)),int(round(1.5*self.S)),int(np.ceil(1/100*self.S)))
        option_prices = [Option(S_values, self.K, self.T, self.r, self.sigma).price_black_scholes()]
        plt.plot(S_values, option_prices)
        plt.title("Black-Scholes Call Option Pricing")
        plt.xlabel("Stock Price ($)")
        plt.ylabel("Call Option Price ($)")
        plt.grid()
        plt.show()

    def price_binomial_tree(self,n=5000):
        dt = self.T / n
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1/u
        p = (np.exp(self.r*dt)-d) / (u - d)
        discount = np.exp(-self.r*dt)
        
        stock_prices_at_maturity = self.S * (u ** np.arange(n, -1, -1)) * (d ** np.arange(0, n + 1, 1))
        if self.option_type == "call":
            option_values = np.maximum(stock_prices_at_maturity-self.K,0)
        else:
            option_values = np.maximum(self.K-stock_prices_at_maturity,0)
        for i in range(n - 1, -1, -1):
            continuation_values = discount * (p * option_values[1:] + (1 - p) * option_values[:-1])
            if self.exercise_type == "american":
                stock_price_at_node = self.S * (u ** np.arange(i, -1, -1)) * (d ** np.arange(0, i + 1, 1))
                if self.option_type == "call":
                    intrinsic_values = np.maximum(stock_price_at_node-self.K,0)
                else:
                    intrinsic_values = np.maximum(self.K-stock_price_at_node,0)
                option_values = np.maximum(continuation_values, intrinsic_values)

            elif self.exercise_type == "european":
                option_values = continuation_values

        return option_values[0]
    
    def price_monte_carlo(self, n):
        if self.exercise_type == "american":
            raise ValueError("Black-Scholes formula is not valid for American options. Use a binomial tree instead.")
        Z = np.random.standard_normal(n)
        ST = self.S * np.exp((self.r - 0.5 * (self.sigma**2)) * self.T + self.sigma * (np.sqrt(self.T) * Z))
        
        if self.option_type == 'call':
            payoffs = np.maximum(ST-self.K,0)
            call = np.sum(payoffs)/n * np.exp(-self.r*self.T)
            return call
        
        elif self.option_type == 'put':
            payoffs = np.maximum(self.K-ST,0)
            put = np.sum(payoffs)/n * np.exp(-self.r*self.T)
            return put
    
    def plot_monte_carlo_convergence(self,max_simulations):
        bs_price = self.price_black_scholes()
        Z = np.random.standard_normal(max_simulations)
        ST = self.S * np.exp((self.r - 0.5 * (self.sigma**2)) * self.T + self.sigma * (np.sqrt(self.T) * Z))
        if self.option_type == "call":
            payoffs = np.maximum(0, ST - self.K)
        else:
            payoffs = np.maximum(0, self.K - ST)
        mc_price = np.cumsum(payoffs * np.exp(-self.r * self.T)) / np.arange(1, max_simulations + 1)
        x_values = np.arange(1,max_simulations+1)
        plt.plot(x_values, mc_price, label = "Monte-Carlo Method Pricing")
        plt.axhline(y=bs_price, color="r", linestyle="--", label="Black-Scholes Model Pricing")
        plt.xlabel("Number of Simulations")
        plt.ylabel(f"{self.option_type.capitalize()} Option Price")
        plt.legend()
        plt.show()
    
    def simulate_price_path(self, steps):
        dt = self.T / steps
        Z = np.random.standard_normal(steps)
        daily_returns = np.exp((self.r - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * Z)
        price_path = np.zeros(steps + 1)
        price_path[0] = self.S
        price_path[1:] = self.S * np.cumprod(daily_returns)
        return price_path

class Hedge:
    def __init__(self, option, stock_path, position="short", transaction_cost=0.005):
        self.option = option
        self.stock_path = stock_path
        self.portfolio = None
        self.position = position.lower()
        if self.position not in ["long","short"]:
            raise ValueError("You can only hold a 'long' or 'short' position on the option")
        self.transaction_cost = transaction_cost ## This is a transaction cost per share

    def delta_hedging(self):
        steps = len(self.stock_path) - 1
        dt = self.option.T / steps
        if self.position == "long":
            position_multiplier = 1
        else:
            position_multiplier = -1
        
        initial_price = self.option.price_black_scholes()
        initial_delta = self.option.get_greeks()["delta"]

        portfolio = pd.DataFrame(index=range(steps+1),columns=[
            "Stock Price", "Shares Held", "Option Value", "Cash", "Portfolio Value", "Profit / Loss" ])

        portfolio.loc[0, "Stock Price"] = self.option.S
        portfolio.loc[0, "Shares Held"] = -position_multiplier * initial_delta

        initial_cost = abs(portfolio.loc[0, "Shares Held"]) * self.transaction_cost
        
        portfolio.loc[0, "Option Value"] = position_multiplier * initial_price
        portfolio.loc[0, "Cash"] = -position_multiplier * initial_price - (portfolio.loc[0, "Shares Held"] * self.option.S ) - initial_cost
        portfolio.loc[0, "Portfolio Value"] = portfolio.loc[0, "Shares Held"] * self.option.S + portfolio.loc[0, "Cash"] + portfolio.loc[0, "Option Value"]
        portfolio.loc[0, "Profit / Loss"] = 0

        interest = np.exp(self.option.r*dt)

        for i in range(1, steps + 1):
            
            time_to_expiry = self.option.T - i*dt
            portfolio.loc[i, "Stock Price"] = self.stock_path[i]

            temp_option = Option(self.stock_path[i], self.option.K, time_to_expiry, self.option.r, self.option.sigma, self.option.option_type)

            if i != steps:
                portfolio.loc[i, "Shares Held"] = -position_multiplier * temp_option.get_greeks()["delta"]
                portfolio.loc[i, "Option Value"] = position_multiplier * temp_option.price_black_scholes()
            else:
                if self.option.option_type == "call":
                    portfolio.loc[i, "Option Value"] = position_multiplier * max(0, portfolio.loc[i, "Stock Price"] - self.option.K)
                else:
                    portfolio.loc[i, "Option Value"] = position_multiplier * max(0, self.option.K - portfolio.loc[i, "Stock Price"])
                portfolio.loc[i, "Shares Held"] = 0
            cost_of_trade = abs(portfolio.loc[i, "Shares Held"] - portfolio.loc[i-1, "Shares Held"]) * self.transaction_cost
            portfolio.loc[i, "Cash"] = portfolio.loc[i-1, "Cash"] * interest - (portfolio.loc[i, "Shares Held"] - portfolio.loc[i-1, "Shares Held"]) * self.stock_path[i] - cost_of_trade
            portfolio.loc[i, "Portfolio Value"] = portfolio.loc[i, "Cash"] + portfolio.loc[i, "Stock Price"] * portfolio.loc[i, "Shares Held"] + portfolio.loc[i, "Option Value"]
            portfolio.loc[i, "Profit / Loss"] = portfolio.loc[i, "Portfolio Value"] - portfolio.loc[0, "Portfolio Value"]

        final_profit_loss = portfolio.loc[steps, "Profit / Loss"]
        self.portfolio = portfolio
        return final_profit_loss
    
    def plot_delta_hedge(self):
        if self.portfolio is None:
            print("Please run the simulation first")
            return
        
        if self.position == "long":
            position_multiplier = 1
        else:
            position_multiplier = -1
    
        portfolio = self.portfolio
        initial_price = self.option.price_black_scholes()
        final_stock_price = portfolio.loc[len(portfolio)-1, "Stock Price"]

        if self.option.option_type == "call":
            final_payout = max(0, final_stock_price - self.option.K)
        else:
            final_payout = max(0, self.option.K - final_stock_price)
        
        unhedged_profit_loss = position_multiplier * (final_payout - initial_price)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        fig.suptitle("Delta Hedging Simulation Results ("+self.position.capitalize()+" "+self.option.option_type.capitalize()+")", fontsize=16)

        ax1.plot(portfolio.index, portfolio["Stock Price"], label="Stock Price Path", color="skyblue")
        ax1.axhline(y=self.option.K, color='r', linestyle='--', label=f"Strike Price (${self.option.K})")
        ax1.set_ylabel('Stock Price ($)')
        ax1.set_title('Simulated Stock Price Path')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        ax2.plot(portfolio.index, portfolio["Profit / Loss"], label="Hedged Portfolio Profit / Loss", color="forestgreen")
        ax2.axhline(y=unhedged_profit_loss, color="darkorange",linestyle="--", label=f'Unhedged Profit/Loss (${unhedged_profit_loss:.2f})')
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Profit and Loss ($)")
        ax2.set_title("Portfolio Profit / Loss: Hedged vs. Unhedged")
        ax2.legend()
        ax2.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

my_option = Option(100,105,1,0.05,0.2, "call")
print(my_option.get_implied_volatility(10))
stock_path = my_option.simulate_price_path(252)
my_hedge = Hedge(my_option, stock_path, "short")
my_hedge.delta_hedging()
my_hedge.plot_delta_hedge()
