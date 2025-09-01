# Derivatives-Lab

## 1. Overview
Welcome! This is a python toolkit for pricing financial options using Black-Scholes, Binomial Tree and Monte-Carlo methods. It calculates the greeks and implied volatility. It also includes a dynamic simulation to visualise and test a delta-hedging strategy for risk management.

## 2. Features

1.    **Option Pricing Models**
Black-Scholes for European Call and Put Options
Monte-Carlo Simulations for European Call and Put Options
Binomial Tree for American and European Call and Put Options

2.   **Greeks Calculation**
Calculates Delta, Gamma, Vega, Theta and Rho

3.   **Implied Volatility**
Solves for implied volatility of a stock based on a given market price of an option

4.   **Delta-Hedging Simulation**
Simulates a delta hedging strategy for holding a short/long position on a call/put option
Visualises the P&L of the hedged portfolio against an unhedged position
Compares the simulated stock price path against the strike price

## 3. Installation

1.   **Clone the repository**

2.   **Install the required modules**
These have been specified in the requirements.txt file in the repository.

3.   **Load the script**
Load the Derivatives-Lab.py file into an IDE (I use VS Code)

## 4. Usage

For pricing options, first create an Option object by specifying, current stock price, strike price, time to expiry, risk-free rate and volatility. Can also add in option type and exercise type, if these are not specified, the option defaults to a european call option. You can then call any of the option pricing functions on the option

In order to simulate a delta-hedge, similarly we need to create a call Option object. We then also need a stock price path, I have included a function to simulate a price path for an option, alternatively a price path can be inputted. We can then create a DeltaHedging object with the option and the price path. Then call the delta hedge and plot delta hedge functions onto the object and this will simulate and visualise a delta-hedging portfolio against an unhedged position for a short call option.

