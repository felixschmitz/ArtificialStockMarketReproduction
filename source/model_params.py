parameters = {
    "experimentSplit": 1,  # binary with 0 for no split and 1 for split
    "batches": 25,  # number of batches for split experiment
    "importPath": r"results/ASM_50000_30082023-133915",  # path to innate rules if mode == 3
    "seed": 42,  # seed for random number generator
    "forecastAdaptation": 0,  # binary with 0 for slow/re and 1 for fast/complex
    "mode": 0,  # with standard mode [0], diagnostics test "clamped" hree predictors [1],
    # diagnostics test hree adaptation [2], and innating rules (pre-trained rules) [3]
    "steps": 2.5e5,  # num of steps/iterations by the model
    "N": 25,  # num of agents & num of assets
    "averageDividend": 10,  # \bar{d} in the paper
    "autoregressiveParam": 0.95,  # \rho in the paper
    "errorVar": 0.0743,  # \sigma_e^2 in the paper
    "dorra": 0.5,  # degree of relative risk aversion; \lambda in the paper
    "interestRate": 0.1,  # r in the paper
    "C": 0.005,  # cost levied for specificity
    "initialPredictorVariance": 4.0,  # \sigma_{p+d}^2 in the paper; h.r.e.e. variance
    "M": 100,  # number of predictors per agent
    "initialCash": 20000,  # initial cash of each agent (cf. Ehrentreich2008 p.94)
    "minCash": 0,  # minimum cash owned by each agent
    "epsilon": 1e-2,  # maximum deviation in specialist's market clearing
    "trialsSpecialist": 10,  # trials per timestep for market clearing
    "maxBid": 10,  # maximum bid of each agent to the risky-asset
    "minPrice": 0.01,  # minimum price for market clearing for specialist
    "maxPrice": 500,  # maximum price for market clearing for specialist
    "minHolding": 0,  # minimum holding for market clearing of each agent
    "hreeA": 0.95,  # h.r.e.e. forecast paramter a
    "hreeB": 4.501,  # h.r.e.e. forecast paramter b
}
