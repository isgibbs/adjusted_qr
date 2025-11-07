import numpy as np
import cvxpy as cp
from cvxpy.error import SolverError

from conditionalconformal import CondConf

### Generate data from a Gaussian linear model
def generateGaussData(nTrain, nTest, d, sigma_sq_eps=1, wZero=False):
    XTrain = np.random.normal(size=(nTrain,d))
    XTest = np.random.normal(size=(nTest,d))
    
    if wZero:
        w = np.zeros(d)
    else:
        w = np.random.normal(size=d)/(d**(1/2))
    
    YTrain = XTrain@w + np.sqrt(sigma_sq_eps)*np.random.normal(size=nTrain)
    YTest = XTest@w +  np.sqrt(sigma_sq_eps)*np.random.normal(size=nTest)
    

    return XTrain, YTrain, XTest, YTest, w

### Compute primal and dual quantile regression solutions
def runQR(X,Y,alpha,lamb=0,intercept=True):
    n = len(Y)
    eta = cp.Variable(n)

    if lamb == 0:
        constraints = [eta@X == 0, -alpha <= eta, eta <= 1-alpha]
        if intercept:
            constraints = [sum(eta) == 0] + constraints
        obj = cp.Maximize(eta@Y)
        prob = cp.Problem(obj,constraints)

        try:
            prob.solve(solver="MOSEK")
        except SolverError as e:
            print('Solver Error: ', (X.shape,alpha,lamb))
            return (0,np.zeros(X.shape[1]),np.zeros(len(X)))
            
        if intercept:
            beta0 = constraints[0].dual_value
            beta = constraints[1].dual_value
        else:
            beta0 = None
            beta = constraints[0].dual_value
    else:
        constraints = [-alpha <= eta, eta <= 1-alpha]
        if intercept:
            constraints = [sum(eta) == 0] + constraints
        obj = cp.Maximize(eta@Y - (1/(4*lamb*n))*cp.sum_squares(X.T@eta))
        prob = cp.Problem(obj,constraints)
        
        try:
            prob.solve(solver="MOSEK")
        except SolverError as e:
            print('Solver Error: ', (X.shape,alpha,lamb))
            return (0,np.zeros(X.shape[1]),np.zeros(len(X)))
            
        if intercept:
            beta0 = constraints[0].dual_value
        else:
            beta0 = None
        beta = (1/(2*lamb*n))*eta.value@X

    return (beta0,beta,eta.value)


### Standard k-fold cross-validation for minimizing quantile loss
def runCV(X,Y,alpha,k,minlamb,maxlamb,numlamb):
    lambs = np.linspace(minlamb,maxlamb,numlamb)
    folds = KFold(n_splits = k, shuffle = True)
               
    allLosses = np.zeros(len(lambs))
    countL = 0
    for lamb in lambs:
        for i, (trainIndex, testIndex) in enumerate(folds.split(X)):

            beta0, beta, eta = runQR(X[trainIndex,:],Y[trainIndex],alpha,lamb)
            
            resid = (Y[testIndex] - beta0 - X[testIndex,:]@beta)
            loss = np.mean(0.5 * np.abs(resid) + (1 - alpha - 0.5)*resid)
    
            allLosses[countL] = allLosses[countL] + loss/k
            
        countL = countL + 1
        
        
    return allLosses, lambs

### Binary search for additive adjustment c with smallest LOO coverage error
def findAddShiftForCoverage(X, Y, alpha, minC = -10, maxC = 10, minCGap = 0.00000001, minCovGap = None, lamb=0):
    if minCovGap is None:
        minCovGap = 1/len(Y)
        
    while maxC - minC > minCGap:
        midC = (minC + maxC)/2

        _, _, eta = runQR(X,Y - midC,alpha,lamb=lamb,intercept=False)
        
        curCov = np.mean(eta <= 0)
        if np.abs(curCov - (1-alpha)) <= minCovGap:
            return midC
        elif curCov > 1-alpha:
            maxC = midC
        else:
            minC = midC
            
    return midC

### Search for the minimum regularization level that gives a LOO coverage of at least 1-alpha-1/n
def findLambForCoverage(X, Y, alpha, minlamb = 0, minCovGap=None, maxlamb = 50, lambIncrement = 0.005):
    if minCovGap is None:
        minCovGap = 1/len(Y)
    
    lamb = minlamb
    while lamb <= maxlamb:
        _, _, eta = runQR(X,Y,alpha,lamb)
        curCov = np.mean(eta <= 0)

        if curCov >= 1-alpha-minCovGap:
            return lamb
        else:
            lamb = lamb + lambIncrement
        
    return lamb

### Binary search for level adjustment with smallest LOO coverage error
def findAdjustedAlpha(X, Y, alpha, lamb, minCovGap=None, minAlphaGap = 0.00000001):
    if minCovGap is None:
        minCovGap = 1/len(Y)
    
    minAlpha = 0
    maxAlpha = 1
    
    while maxAlpha - minAlpha > minAlphaGap:
        alphaAdj = (maxAlpha+minAlpha)/2
        _, _, eta = runQR(X,Y,alphaAdj,lamb)
        
        curCov = np.mean(eta <= 0)
        if np.abs(curCov - (1-alpha)) <= minCovGap:
            return alphaAdj
        elif curCov > 1-alpha:
            minAlpha = alphaAdj
        else:
            maxAlpha = alphaAdj
            
    return alphaAdj

### Compute coverage, length, and multiaccuracy error of fitted upper and lower quantile estimates
def coverageMeasTwoSided(XTest,YTest,beta0Up,betaUp,beta0Down,betaDown,alpha):
    qhatUp = beta0Up + XTest@betaUp
    qhatDown = beta0Down + XTest@betaDown
    
    coveragesUp = YTest <= qhatUp
    coveragesDown = YTest >= qhatDown
    coverages = coveragesUp & coveragesDown
    
    meanCov = np.mean(coverages)
    meanCovUp = np.mean(coveragesUp)
    meanCovDown = np.mean(coveragesDown)
    maCov = (1/len(YTest))*np.max(np.abs((coverages - (1-alpha))@XTest)/np.mean(np.abs(XTest),0))
    maCovUp = (1/len(YTest))*np.max(np.abs((coveragesUp - (1-alpha/2))@XTest)/np.mean(np.abs(XTest),0))
    maCovDown = (1/len(YTest))*np.max(np.abs((coveragesDown - (1-alpha/2))@XTest)/np.mean(np.abs(XTest),0))
    medLength = np.median(np.maximum(qhatUp - qhatDown,0))

    return (meanCov, maCov, medLength, meanCovUp, meanCovDown, maCovUp, maCovDown)

### Compute coverage, cutoff size, and multiaccuracy error of fitted quantile estimate
def coverageMeasurements(XTest,YTest,beta0,beta,alpha):
    qhat = beta0 + XTest@beta
    coverages = (YTest <= qhat)
    
    meanCov = np.mean(coverages)
    maCov = (1/len(YTest))*np.max(np.abs((coverages - (1-alpha))@XTest)/np.mean(np.abs(XTest),0))
    meanLength = np.median(qhat)

    return (meanCov, maCov, meanLength)

### Tune the level and regularization to achieve coverage while minimizing multiaccuracy error
def levelAdjWRegTuning(X,Y,alpha,minlamb,maxlamb,numlamb):
    minCovGap = 1/len(Y)
    
    lambs = np.linspace(minlamb,maxlamb,numlamb)
    curBestMAError = float('inf')
    bestAlpha, bestLamb = (alpha,0)
    for lamb in lambs:
        alphaAdj = findAdjustedAlpha(X, Y, alpha, lamb)
        
        _, _, eta = runQR(X, Y, alphaAdj, lamb)
        if np.abs(np.mean(eta<=0) - (1-alpha)) <= minCovGap:
            maCov = np.max(np.abs(((eta <= 0) - (1-alpha))@X)/np.mean(np.abs(X),0))

            if maCov < curBestMAError:
                bestAlpha, bestLamb = (alphaAdj, lamb)
                curBestMAError = maCov

    return (bestAlpha, bestLamb)

### Tune the additive adjustment and regularization to achieve coverage while minimizing multiaccuracy error
def addAdjWRegTuning(X,Y,alpha,minlamb,maxlamb,numlamb):
    minCovGap = 1/len(Y)
    lambs = np.linspace(minlamb,maxlamb,numlamb)
    curBestMAError = float('inf')
    bestC, bestLamb = (0,0)
    for lamb in lambs:
        cAdj = findAddShiftForCoverage(X, Y, alpha, lamb=lamb)
        
        _, _, eta = runQR(X, Y-cAdj, alpha, lamb, intercept=False)
        if np.abs(np.mean(eta<=0) - (1-alpha)) <= minCovGap:
            maCov = np.max(np.abs(((eta <= 0) - (1-alpha))@X)/np.mean(np.abs(X),0))

            if maCov < curBestMAError:
                bestC, bestLamb = (cAdj, lamb)
                curBestMAError = maCov

    return (bestC, bestLamb)
