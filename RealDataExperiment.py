import numpy as np
import cvxpy as cp
from tqdm import tqdm

from qr_methods import runQR, levelAdjWRegTuning, addAdjWRegTuning, coverageMeasTwoSided
from conditionalconformal import CondConf


def subsetData(X,Y,nTrain,nTest,d):
    cols = np.random.choice(X.shape[1], size=d, replace=False)

    trainSamps = np.random.choice(X.shape[0], size=nTrain, replace=False)

    remaining = np.setdiff1d(np.arange(X.shape[0]), trainSamps)
    testSamps = np.random.choice(remaining, size=nTest, replace=False)

    XTrain, YTrain = X[trainSamps,:][:,cols], Y[trainSamps]
    XTest, YTest   = X[testSamps,:][:,cols], Y[testSamps]

    return (XTrain, YTrain, XTest, YTest)

def oneRealExperiment(X, Y, nTrain, nTest, d, alpha, minC, maxC, minlamb, maxlamb, numlamb, calFrac=0.25, runCC=True):
    
    XTrain, YTrain, XTest, YTest = subsetData(X, Y, nTrain, nTest, d)

    ### Vanilla QR 
    beta0QRUp, betaQRUp, etaQRUp = runQR(XTrain,YTrain,alpha/2,0)
    beta0QRDown, betaQRDown, etaQRDown = runQR(XTrain,YTrain,1-alpha/2,0)
    
    covQR, maQR, lengthQR, covUpQR, covDownQR, maUpQR, maDownQR = coverageMeasTwoSided(XTest,YTest,beta0QRUp, betaQRUp, beta0QRDown, betaQRDown,alpha)

    ### Addditve adjustment with regularization
    cMASUp, lambAddMASUp = addAdjWRegTuning(XTrain,YTrain,alpha/2,minlamb,maxlamb,numlamb)
    cMASDown, lambAddMASDown = addAdjWRegTuning(XTrain,YTrain,1-alpha/2,minlamb,maxlamb,numlamb)
    
    _, betaAddMASUp, _ = runQR(XTrain,YTrain-cMASUp,alpha/2,lambAddMASUp,intercept=False)
    _, betaAddMASDown, _ = runQR(XTrain,YTrain-cMASDown,1-alpha/2,lambAddMASDown,intercept=False)

    covAddAdj, maCovAddAdj, lengthAddAdj, covUpAddAdj, covDownAddAdj, maUpAddAdj, maDownAddAdj = coverageMeasTwoSided(XTest,YTest,cMASUp,betaAddMASUp,cMASDown,betaAddMASDown,alpha)
    

    ### Adjusted level with regularization
    alphaMASUp, lambLevelMASUp = levelAdjWRegTuning(XTrain,YTrain,alpha/2,minlamb,maxlamb,numlamb)
    alphaMASDown, lambLevelMASDown = levelAdjWRegTuning(XTrain,YTrain,1-alpha/2,minlamb,maxlamb,numlamb)

    beta0LevelMASUp, betaLevelMASUp, _ = runQR(XTrain,YTrain,alphaMASUp,lambLevelMASUp)
    beta0LevelMASDown, betaLevelMASDown, _ = runQR(XTrain,YTrain,alphaMASDown,lambLevelMASDown)
    covLevelAdj, maCovLevelAdj, lengthLevelAdj, covUpLevelAdj, covDownLevelAdj, maUpLevelAdj, maDownLevelAdj = coverageMeasTwoSided(XTest,YTest,beta0LevelMASUp,betaLevelMASUp,
                                                                      beta0LevelMASDown,betaLevelMASDown,alpha)


    ### CQR
    nCal = int(calFrac*nTrain)
    XCal = XTrain[0:nCal,:]
    YCal = YTrain[0:nCal]
    XTTrain = XTrain[nCal:,:]
    YTTrain = YTrain[nCal:]
    
    beta0CQRUp, betaCQRUp, _ = runQR(XTTrain,YTTrain,alpha/2)
    beta0CQRDown, betaCQRDown, _ = runQR(XTTrain,YTTrain,1-alpha/2)
    offsetCQRUp = np.quantile(YCal - beta0CQRUp - XCal@betaCQRUp, 1-alpha/2)
    offsetCQRDown = np.quantile(YCal - beta0CQRDown - XCal@betaCQRDown, alpha/2)


    covCQR, maCovCQR, lengthCQR, covUpCQR, covDownCQR, maUpCQR, maDownCQR = coverageMeasTwoSided(XTest,YTest,beta0CQRUp + offsetCQRUp,betaCQRUp,
                                                                      beta0CQRDown + offsetCQRDown,betaCQRDown,alpha)


    res = pd.DataFrame({'Method': ['Quantile Regression', 'Additive Adjustment with Regularization', 'Level Adjustment with Regularization',
                                   'Conformalized Quantile Regression (Romano et al. (2019))'],
                        'Coverage': [covQR, covAddAdj, covLevelAdj, covCQR],
                        'Coverage Up': [covUpQR, covUpAddAdj, covUpLevelAdj, covUpCQR],
                        'Coverage Down': [covDownQR, covDownAddAdj, covDownLevelAdj, covDownCQR],
                        'Length': [lengthQR, lengthAddAdj, lengthLevelAdj, lengthCQR],
                        'Multiaccuracy Error': [maQR, maCovAddAdj, maCovLevelAdj, maCovCQR],
                        'Multiaccuracy Error Up': [maUpQR, maUpAddAdj, maUpLevelAdj, maUpCQR],
                        'Multiaccuracy Error Down': [maDownQR, maDownAddAdj, maDownLevelAdj, maDownCQR],
                        'Dimension': [d]*4,
                        'Number of Training Points': [nTrain]*4,
                        'Lambda' : [(0,0,0,0),(lambAddMASUp,lambAddMASDown,cMASUp,cMASDown),(lambLevelMASUp,lambLevelMASDown,alphaMASUp,alphaMASDown),(0,0,0,0)]
    })

    if runCC:
        ### Fixed dual thresholding
        XTrainInt = np.column_stack((np.ones(nTrain),XTrain))
        XTestInt = np.column_stack((np.ones(nTest),XTest))
        
        scoreFn = lambda x, y : y if isinstance(y, np.ndarray) else np.array([y])
        phiFn = lambda x : x if isinstance(x, np.ndarray) else np.array([x])
        condCovProgram = CondConf(scoreFn, phiFn)
        condCovProgram.setup_problem(XTrainInt,YTrain)
    
        etaCutoffUp = np.quantile(etaQRUp,1-alpha/2)
        etaCutoffDown = np.quantile(etaQRDown,alpha/2)
        covEtaCutUp = 0
        covEtaCutDown = 0
        covEtaCut = 0
        maEtaCutUpAll = np.zeros(XTest.shape[1])
        maEtaCutDownAll = np.zeros(XTest.shape[1])
        maEtaCut = np.zeros(XTest.shape[1])
        allLengthsEtaCut = np.zeros(nTest)
        for i in range(nTest):
            y_cutoff_up = condCovProgram.predict(1-alpha/2, XTestInt[i,:].reshape(1,d+1),
                                            lambda a, b : a, exact=True, threshold = etaCutoffUp)[0]
            y_cutoff_down = condCovProgram.predict(alpha/2, XTestInt[i,:].reshape(1,d+1),
                                            lambda a, b : a, exact=True, threshold = etaCutoffDown)[0]
            covEtaCutUp = covEtaCutUp + (YTest[i] <= y_cutoff_up)/nTest
            covEtaCutDown = covEtaCutDown + (YTest[i] >= y_cutoff_down)/nTest
            covEtaCut = covEtaCut + ((YTest[i] <= y_cutoff_up) and (YTest[i] >= y_cutoff_down))/nTest
            maEtaCutUpAll = maEtaCutUpAll + ((YTest[i] <= y_cutoff_up) - (1-alpha/2))*XTest[i,:]/nTest
            maEtaCutDownAll = maEtaCutDownAll + ((YTest[i] >= y_cutoff_down) - (1-alpha/2))*XTest[i,:]/nTest
            maEtaCut = maEtaCut + ((YTest[i] <= y_cutoff_up) and (YTest[i] >= y_cutoff_down) - (1-alpha))*XTest[i,:]/nTest
            allLengthsEtaCut[i] = np.max((y_cutoff_up - y_cutoff_down),0)
        lengthEtaCut = np.median(allLengthsEtaCut)
        maUpEtaCut = np.max(np.abs(maEtaCutUpAll)/np.mean(np.abs(XTest),0))
        maDownEtaCut = np.max(np.abs(maEtaCutDownAll)/np.mean(np.abs(XTest),0))
        maCovEtaCut = np.max(np.abs(maEtaCut)/np.mean(np.abs(XTest),0))
    
        ### Randomized method of Gibbs, Cherian, and Candes (2025)
        covCCUp = 0
        covCCDown = 0
        covCC = 0
        maCCUpAll = np.zeros(XTest.shape[1])
        maCCDownAll = np.zeros(XTest.shape[1])
        maAllCovCC = np.zeros(XTest.shape[1])
        allLengthsCC = np.zeros(nTest)
        for i in range(nTest):
            y_cutoff_up = condCovProgram.predict(1-alpha/2, XTestInt[i,:].reshape(1,d+1),
                                            lambda a, b : a, exact=True, randomize=True)[0]
            y_cutoff_down = condCovProgram.predict(alpha/2, XTestInt[i,:].reshape(1,d+1),
                                            lambda a, b : a, exact=True, randomize=True)[0]
            covCCUp = covCCUp + (YTest[i] <= y_cutoff_up)/nTest
            covCCDown = covCCDown + (YTest[i] >= y_cutoff_down)/nTest
            covCC = covCC + ((YTest[i] <= y_cutoff_up) and (YTest[i] >= y_cutoff_down))/nTest
            maCCUpAll = maCCUpAll + ((YTest[i] <= y_cutoff_up) - (1-alpha/2))*XTest[i,:]/nTest
            maCCDownAll = maCCDownAll + ((YTest[i] >= y_cutoff_down) - (1-alpha/2))*XTest[i,:]/nTest
            maAllCovCC = maAllCovCC + ((YTest[i] <= y_cutoff_up) and (YTest[i] >= y_cutoff_down) - (1-alpha))*XTest[i,:]/nTest
            allLengthsCC[i] = np.max((y_cutoff_up - y_cutoff_down),0)
        lengthCC = np.median(allLengthsCC)
        maUpCovCC = np.max(np.abs(maCCUpAll)/np.mean(np.abs(XTest),0))
        maDownCovCC = np.max(np.abs(maCCDownAll)/np.mean(np.abs(XTest),0))
        maCovCC = np.max(np.abs(maAllCovCC)/np.mean(np.abs(XTest),0))

        res = pd.concat([res,pd.DataFrame({'Method': ['Fixed Dual Thresholding', 'Randomized Dual Thresholding (Gibbs et al. (2025))'],
                        'Coverage': [covEtaCut, covCC],
                        'Coverage Up': [covEtaCutUp, covCCUp],
                        'Coverage Down': [covEtaCutDown, covCCDown],
                        'Length': [lengthEtaCut, lengthCC],
                        'Multiaccuracy Error': [maCovEtaCut, maCovCC],
                        'Multiaccuracy Error Up': [maUpEtaCut, maUpCovCC],
                        'Multiaccuracy Error Down': [maDownEtaCut, maDownCovCC],
                        'Dimension': [d]*2,
                        'Number of Training Points': [nTrain]*2,
                        'Lambda': [(0,0,0,0)]*2
        })])


    return res


DEFAULTS = dict(
    nTrials=20,
    nTrains=[200],
    nTest=2000,
    ds= [1,10,20,30,40,50],
    alpha=0.1,
    minlamb=0.0,
    maxlamb=0.2,
    numlamb=41,
    minC=-10,
    maxC=10,
    gamma=0.1,
    calFrac=0.25,
    dataset='News'
)


import time, math, argparse
from itertools import product
from multiprocessing import Pool

### One experiment at a specific set of parameter settings
def _worker(task):
    # task = (trialID, d, nTrain, params)
    _, d, nTrain, P = task
    res = oneRealExperiment(P["X"], P["Y"], nTrain, P["nTest"], d, P["alpha"], P["minC"], P["maxC"],
        P["minlamb"], P["maxlamb"], P["numlamb"], P["calFrac"]
    )
    return res

### Run many trials in parallel
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
    ap.add_argument("--calFrac", type=float, default=DEFAULTS["calFrac"])
    ap.add_argument("--dims", type=int, nargs="+", default=DEFAULTS["ds"])
    ap.add_argument("--n-jobs", type=int, default=8)            # how many parallel processes
    ap.add_argument("--dataset", type=str, default=DEFAULTS["dataset"])
    ap.add_argument("--out", type=str, default="NewsResults.csv")
    args = ap.parse_args()

    if args.dataset == 'CandC':
        attrib = pd.read_csv('data/CandCData/attributes.csv', sep='\s+')
        data = pd.read_csv('data/CandCData/communities.data', names = attrib['attributes'])
        
        numeric = data.select_dtypes(include=[np.number]).columns.tolist()
        target = "ViolentCrimesPerPop"
        numeric.remove(target)
        numeric.remove('state')
        numeric.remove('fold')
        
        XRaw = data[numeric].values
        YRaw = data[target].values
    elif args.dataset == 'Super':
        data = pd.read_csv('data/superconductivty+data/train.csv').to_numpy()
        
        XRaw = data[:,0:(data.shape[1]-1)]
        YRaw = data[:,-1]
    elif args.dataset == 'News':
        data = pd.read_csv('data/OnlineNewsPopularity/OnlineNewsPopularity.csv')
        
        data = data.iloc[:,1:].to_numpy()
        XRaw = data[:,0:(data.shape[1]-1)]

        XRaw = XRaw[:,[j for j in range(XRaw.shape[1]) if j not in [36, 37, 48, 59]]] 
        YRaw = data[:,-1]

        
    # drop zero-variance, standardize
    Y = (YRaw - YRaw.mean())/YRaw.std()
    std = XRaw.std(axis=0)
    XRaw = XRaw[:, std > 0]
    X = (XRaw - XRaw.mean(axis=0)) / XRaw.std(axis=0)


        

    P = dict(
        nTest=args.n_test, alpha=args.alpha,
        minlamb=args.minlamb, maxlamb=args.maxlamb, numlamb=args.numlamb,
        minC=args.minC, maxC=args.maxC, calFrac=args.calFrac, X=X, Y=Y
    )

    ### Build all tasks 
    if len(args.n_trains) == 1:
        tasks = [(i, d, args.n_trains[0], P) for i, d in product(range(args.n_trials), args.dims)]
    else:
        tasks = []
        for n in args.n_trains:
            d = int(args.gamma*n)
            tasks += [(i, d, n, P) for i in range(args.n_trials)]
    total = len(tasks)

    ### Run tasks in parallel
    results = []
    with Pool(processes=args.n_jobs) as pool:
        for df in tqdm(pool.imap_unordered(_worker, tasks), total=total, desc="Running experiments"):
            results.append(df)


    res = pd.concat(results, ignore_index=True)

    ### Save Results
    res.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()



