# Reproduction of the Santa Fe Institute Artificial Stock Market (SFI-ASM)
## Scientific Source and Context
The original paper of the SFI-ASM has been reproduced by myself as project for my Bachelor thesis for the degree of B.A. Philosophy & Economics at the University of Bayreuth under the supervision of [Dr. Paolo Galeazzi](https://www.phil.uni-bayreuth.de/en/people/galeazzi/index.php).

The reference to the paper is: <br>
Arthur, W. Brian and Holland, John H. and LeBaron, Blake D. and Palmer, Richard G. and Tayler, Paul, Asset Pricing Under Endogenous Expectations in an Artificial Stock Market (Dec 12, 1996). Available at SSRN: https://ssrn.com/abstract=2252 or http://dx.doi.org/10.2139/ssrn.2252

## Model Overview
The SFI-ASM is an agent-based artifical stock market model in which agent's co-create a market for a risky asset by trading it. The model is based on the assumption that agents are inductively rational and therefore do not have information about the expectations of others. In this sense the model is heterogeneous. They form expectations about the future price of the asset based on past prices. The model is able to reproduce stylized facts of real-world financial markets, such as fat tails in the distribution of returns, volatility clustering and long-term memory in the volatility of returns.

## Model Description
The model is implemented in Python 3.11.0. The main file is `run.py`. The model is run by executing this file with the parameters specified in `model_params.py`. One `ap.Model` object gets initilized as `ArtificialStockMarket` and contains the global artificial market specific procedures. Within this `ArtificialStockMarket`, `ap.Agent` objects get initialized as `MarketStatistician` and contain the agent specific procedures. The `ArtificialStockMarket` object contains a list of `MarketStatistician` objects. 

The `ArtificialStockMarket` object contains the following procedures:
- `setup()`: sets up the model by initializing the agents and the market
- `step()`: executes one step of the model, i.e. one time period
- `document()`: documents the model by saving the data of the agents and the market for the current time period
- `update()`: updates the model by updating the agents and the market for the current time period

The `MarketStatistician` object contains the following procedures:
- `setup()`: sets up the agent by initializing the agent's variables
- `step()`: executes one step of the agent, i.e. one time period
- `document()`: documents the agent by saving the agent's variables for the current time period
- `update()`: updates the agent by updating the agent's variables for the current time period
- `end()`: returns specific agent's variables for the last time period

## Model Parameters
The model parameters are specified in `model_params.py`. The main parameters are:
- `N`: number of agents initially endowned with one risky asset
- `steps`: number of time periods
- `averageDividend`: average dividend for the AR(1) process of the risky asset
- `experimentSplit`: binary with 0 for no split and 1 for split of the experiment into multiple batches

## Model Output
The model output is saved in the `results` folder either in batches or as one directory depending on the `experimentSplit` parameter.

## Model Reproduction
The model can be reproduced by reproducing the conda env from the `req.yml` and executing `run.py` which takes the parameters specified in `model_params.py`. One `ap.Model` object gets initilized as `ArtificialStockMarket` and contains the global artificial market specific procedures. Within this `ArtificialStockMarket`, `ap.Agent` objects get initialized as `MarketStatistician` and contain the agent specific procedures.