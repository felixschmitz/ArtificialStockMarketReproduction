parameters = {
    "importPath": r"results/ASM_100000_12072023-160700",  # path to innate rules
    "seed": 42,
    "forecastAdaptation": 1,  # binary with 0 for slow and 1 for fast
    "mode": 3,  # with standard mode [0], diagnostics test hree [1],
    # diagnostics test adaptation [2], and innating rules (pre-trained rules) [3]
    # 'dividendSequence': None, # for mode 2; diagnostics test with given dividend sequence
    # 'priceSeries': None, # for mode 2; diagnostics test with given price series
    "steps": 1e2,  # num of steps/iterations by the model
    "N": 25,  # num of agents & num of assets
    "averageDividend": 10,  # \bar{d}
    "autoregressiveParam": 0.95,  # in the paper rho
    "errorVar": 0.0743,
    "dorra": 0.5,  # degree of relative risk aversion
    "interestRate": 0.1,
    "C": 0.005,  # cost levied for specificity
    "initialPredictorVariance": 4.0,  # 3.999769641,
    "M": 100,  # number of predictors per agent
    "initialCash": 20000,  # initial cash of each agent in the bank (cf. Ehrentreich2008 p.94)
    "minCash": 0,  # minimum cash of each agent in the bank
    "trialsSpecialist": 10,  # trials per timestep for market clearing
    "specialistType": 1,  # 0: ration expectations, 1: slope, 2: fixed ETA; default: 1
    "minExcess": 0.01,  # minimum excess demand for market clearing
    "minPrice": 0.01,  # minimum price for market clearing
    "maxPrice": 500,  # maximum price for market clearing
    "minHolding": -5,  # minimum holding for market clearing
    "maxBid": 10,  # maximum nummber of asset bid of each agent
}
