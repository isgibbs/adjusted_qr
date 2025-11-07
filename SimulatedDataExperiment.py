import numpy as np
import cvxpy as cp
from tqdm import tqdm
import pandas as pd

from qr_methods import generateGaussData, runQR, levelAdjWRegTuning, addAdjWRegTuning
from conditionalconformal import CondConf

### Evaluate coverage of various methods on simulated data from a Gaussian linear model
def oneSimExperiment(nTrain, nTest, d, alpha, minC, maxC, minlamb, maxlamb, numlamb):
    
    XTrain, YTrain, XTest, YTest, _ = generateGaussData(nTrain, nTest, d)

    ### Vanilla QR 
    beta0QR, betaQR, etaQR = runQR(XTrain,YTrain,alpha,0)
    covQR = np.mean(YTest <= beta0QR + XTest@betaQR)

    ### Fixed dual thresholding 
    XTrainInt = np.column_stack((np.ones(nTrain),XTrain))
    XTestInt = np.column_stack((np.ones(nTest),XTest))
    
    scoreFn = lambda x, y : y if isinstance(y, np.ndarray) else np.array([y])
    phiFn = lambda x : x if isinstance(x, np.ndarray) else np.array([x])
    condCovProgram = CondConf(scoreFn, phiFn)
    condCovProgram.setup_problem(XTrainInt,YTrain)

    etaCutoff = np.quantile(etaQR,1-alpha)
    covEtaCut = 0
    for i in range(nTest):
        fdQuantEst = condCovProgram.predict(1-alpha, XTestInt[i,:].reshape(1,d+1),
                                        lambda a, b : a, exact=True, threshold = etaCutoff)[0]
        covEtaCut = covEtaCut + (YTest[i] <= fdQuantEst)/nTest

    ### Addditve adjustment with regularization
    cMAS, lambAddMAS = addAdjWRegTuning(XTrain,YTrain,alpha,minlamb,maxlamb,numlamb)
    _, betaAddMAS, _ = runQR(XTrain,YTrain-cMAS,alpha,lambAddMAS,intercept=False)
    covAddAdj = np.mean(YTest <= cMAS + XTest@betaAddMAS)

    ### Adjusted level with regularization
    alphaMAS, lambLevelMAS = levelAdjWRegTuning(XTrain,YTrain,alpha,minlamb,maxlamb,numlamb)
    beta0LeveMAS, betaLevelMAS, _ = runQR(XTrain,YTrain,alphaMAS,lambLevelMAS)
    covLevelAdj = np.mean(YTest <= beta0LeveMAS + XTest@betaLevelMAS)

    return (covQR, covEtaCut, covAddAdj, covLevelAdj)


DEFAULTS = dict(
    nTrials=100,
    nTrains=[200],
    nTest=2000,
    ds=[1,10,20,30,40,50],
    alpha=0.1,
    minlamb=0.0,
    maxlamb=0.1,
    numlamb=21,
    minC=-10,
    maxC=10,
    gamma=0.1
)


import time, math, argparse
from itertools import product
from multiprocessing import Pool

### one simulated experiment at a specific set of parameter settings
def _worker(task):
    # task = (trialID, d, nTrain, params)
    _, d, nTrain, P = task
    covQR, covEtaCut, covAddAdj, covLevelAdj = oneSimExperiment(
        nTrain, P["nTest"], d, P["alpha"], P["minC"], P["maxC"],
        P["minlamb"], P["maxlamb"], P["numlamb"]
    )
    return pd.DataFrame({
        "Coverage": [covQR, covEtaCut, covAddAdj, covLevelAdj],
        "Method": [
            "Quantile Regression",
            "Fixed Dual Thresholding",
            "Additive Adjustment with Regularization",
            "Level Adjustment with Regularization",
        ],
        "Dimension": [d]*4,
        "Number of Training Points": [nTrain]*4
    })

### run many simulated experiments in parallel
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=DEFAULTS["nTrials"])
    ap.add_argument("--n-trains", type=int, nargs="+", default=DEFAULTS["nTrains"])
    ap.add_argument("--n-test", type=int, default=DEFAULTS["nTest"])
    ap.add_argument("--gamma", type=float, default=DEFAULTS["gamma"])
    ap.add_argument("--alpha", type=float, default=DEFAULTS["alpha"])
    ap.add_argument("--minlamb", type=float, default=DEFAULTS["minlamb"])
    ap.add_argument("--maxlamb", type=float, default=DEFAULTS["maxlamb"])
    ap.add_argument("--numlamb", type=int, default=DEFAULTS["numlamb"])
    ap.add_argument("--minC", type=float, default=DEFAULTS["minC"])
    ap.add_argument("--maxC", type=float, default=DEFAULTS["maxC"])
    ap.add_argument("--dims", type=int, nargs="+", default=DEFAULTS["ds"])
    ap.add_argument("--n-jobs", type=int, default=4)            # how many processes
    ap.add_argument("--out", type=str, default="simulatedResults.csv")
    args = ap.parse_args()

    P = dict(
        nTest=args.n_test, alpha=args.alpha,
        minlamb=args.minlamb, maxlamb=args.maxlamb, numlamb=args.numlamb,
        minC=args.minC, maxC=args.maxC
    )

    # build all tasks to run in parallel
    if len(args.n_trains) == 1:
        tasks = [(i, d, args.n_trains[0], P) for i, d in product(range(args.n_trials), args.dims)]
    else:
        tasks = []
        for n in args.n_trains:
            d = int(args.gamma*n)
            tasks += [(i, d, n, P) for i in range(args.n_trials)]
    total = len(tasks)

    results = []

    # run tasks
    with Pool(processes=args.n_jobs) as pool:
        for df in tqdm(pool.imap_unordered(_worker, tasks), total=total, desc="Running experiments"):
            results.append(df)


    res = pd.concat(results, ignore_index=True)

    # save results
    res.to_csv(args.out, index=False)
    print(f"Saved dataframe with shape {res.shape} to: {args.out}")


if __name__ == "__main__":
    main()
