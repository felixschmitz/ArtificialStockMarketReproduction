parameters = {
    "experimentSplit": 1,  # binary with 0 for no split and 1 for split
    "importPath": r"results/ASM_50000_30082023-133915",  # path to innate rules
    "seed": 42,  # seed for random number generator
    "forecastAdaptation": 0,  # binary with 0 for slow/hree and 1 for fast/complex
    "mode": 0,  # with standard mode [0], diagnostics test "clamped" hree predictors [1],
    # diagnostics test hree adaptation [2], and innating rules (pre-trained rules) [3]
    "steps": 2.5e5,  # num of steps/iterations by the model
    "N": 25,  # num of agents & num of assets
    "averageDividend": 10,  # \bar{d}
    "autoregressiveParam": 0.95,  # in the paper rho
    "errorVar": 0.0743,
    "dorra": 0.5,  # degree of relative risk aversion
    "interestRate": 0.1,  # in the paper r
    "C": 0.005,  # cost levied for specificity
    "initialPredictorVariance": 4.0,  # see paper,
    "M": 100,  # number of predictors per agent
    "initialCash": 20000,  # initial cash of each agent in the bank (cf. Ehrentreich2008 p.94)
    "minCash": 0,  # minimum cash of each agent in the bank
    "epsilon": 1e-2,  # maximum deviation in specialist's market clearing
    "trialsSpecialist": 10,  # trials per timestep for market clearing
    "maxBid": 10,  # maximum bid of each agent to the risky-asset
    "minPrice": 0.01,  # minimum price for market clearing
    "maxPrice": 500,  # maximum price for market clearing
    "minHolding": 0,  # minimum holding for market clearing
    "hreeA": 0.95,  # hree forecast paramter a
    "hreeB": 4.501,  # hree forecast paramter b
}
