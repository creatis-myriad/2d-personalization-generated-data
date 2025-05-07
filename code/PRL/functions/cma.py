import numpy as np
import time
from functions.generations import generate_elliptical, generate_spherical
from functions.loss import loss_function

#adapted from the minimalistic Matlab implementation http://www.cmap.polytechnique.fr/~nikolaus.hansen/cmaes_inmatlab.html#matlab
def cmaes2D(optionGeneration,optionLoss,numberCases,numberParam,xStart,sigmaStart,stopFitness,stopEval,lmbda,XR,XRLatent,indMyocardium,knn,X0,Y0,X,Y,startZone,myocardium,numberPixels,coords):  
    #parameters
    start = time.time()
    xmean = xStart 
    sigma = sigmaStart
    stopfitness = stopFitness  #stop if fitness < stopfitness
    stopeval = stopEval   #stop after stopeval number of function evaluations
    mu = int(np.floor(lmbda/4))
    weights = np.log((lmbda+1)/2)-np.log(np.linspace(1,mu,mu))
    weights = weights/np.sum(weights)
    mueff=np.sum(weights)**2/np.sum(weights**2)

    #strategy parameter setting: adaptation
    cc = (4+mueff/numberParam)/(numberParam+4+2*mueff/numberParam)
    cs = (mueff+2)/(numberParam+mueff+5)
    c1 = 2/((numberParam+1.3)**2+mueff)
    cmu = min(1-c1,2*(mueff-2+1/mueff)/((numberParam+2)**2+mueff))
    damps = 1 + 2*max(0,np.sqrt((mueff-1)/(numberParam+1))-1)+cs 
    
    #initialize dynamic (internal) strategy parameters and constants
    pc = np.zeros(numberParam)
    ps = np.zeros(numberParam)
    B = np.eye(numberParam,numberParam)                   
    D = np.ones(numberParam)
    C = B@np.diag(D**2)@B.T
    invsqrtC = B@np.diag(D**(-1))@B.T
    eigeneval = 0
    chiN=numberParam**0.5*(1-1/(4*numberParam)+1/(21*numberParam**2))
    
    #for output
    outdatx = []
    outdata = []
    outsigma = []
    outd = []

    #generation loop
    counteval = 0
    while (counteval<stopeval):
        if optionGeneration==1:
            arx=np.zeros((numberParam,lmbda))
            arfitness=np.zeros(lmbda)
            for k in range(lmbda):
                arx[:,k] = xmean+sigma*B@np.multiply(D,np.random.normal(0,1,numberParam))
                arfitness[k] = loss_function(optionGeneration,optionLoss,arx[:,k],numberCases,XR,XRLatent,indMyocardium,knn,X0,Y0,X,Y,startZone,myocardium,numberPixels,coords)
                counteval = counteval+1
        else:
            arx=np.zeros((numberParam,lmbda))
            arfitness=np.zeros(lmbda)
            for k in range(lmbda):
                arx[:,k] = xmean+sigma*B@np.multiply(D,np.random.normal(0,1,numberParam))
                while np.any(arx[:,k] < 0):
                    arx[:,k] = xmean+sigma*B@np.multiply(D,np.random.normal(0,1,numberParam))
                arfitness[k] = loss_function(optionGeneration,optionLoss,arx[:,k],numberCases,XR,XRLatent,indMyocardium,knn,X0,Y0,X,Y,startZone,myocardium,numberPixels,coords)
                counteval = counteval+1

        #sort by fitness and compute weighted mean into xmean
        arindex = np.argsort(arfitness)
        arfitness = np.sort(arfitness)
        xold = xmean
        tmp = np.zeros((numberParam,mu))
        for k in range(mu):
            ind = arindex[k]
            tmp[:,k] = arx[:,ind]
        xmean = tmp@weights  #recombination, new mean value
        
        #cumulation: update evolution paths
        ps = (1-cs)*ps+np.sqrt(cs*(2-cs)*mueff)*invsqrtC@(xmean-xold)/sigma
        hsig = np.sum(ps**2)/(1-(1-cs)**(2*counteval/lmbda))/numberParam < 2+4/(numberParam+1)
        pc = (1-cc)*pc+hsig*np.sqrt(cc*(2-cc)*mueff)*(xmean-xold)/sigma

        #adapt covariance matrix C
        repmatxold = np.zeros(tmp.shape)
        for j in range(tmp.shape[1]):
            repmatxold[:,j] =  xold
        artmp = (1/sigma)*(tmp-repmatxold)
        C = (1-c1-cmu)*C+c1*(pc@pc.T+(1-hsig)*cc*(2-cc)*C)+cmu*artmp@np.diag(weights)@artmp.T

        #adapt step size sigma
        sigma = sigma*np.exp((cs/damps)*(np.linalg.norm(ps)/chiN-1))
        
        #update B and D from C
        if counteval-eigeneval>lmbda/(c1+cmu)/numberParam/10: 
            eigeneval = counteval
            C = np.triu(C)+np.triu(C,1).T
            D, B = np.linalg.eig(C)
            D = np.sqrt(D)
            invsqrtC = B@np.diag(D**(-1))@B.T 

        #break, if fitness is good enough
        if (arfitness[0]<=stopfitness) or (np.max(D)>1e7 * np.min(D)):
            print('break1')
            break
            

        outdata.append(arfitness[0])
        outsigma.append(sigma)
        outd.append(D)
        outdatx.append(xmean)
        current = time.time()
        passed = current - start
        print('Passed time: ', passed, ' Function evaluations: ', counteval)
                
                
    #final Message
    if optionGeneration==1:
        print('Iterations: ', xmean[0], '  Max radius: ', xmean[1])
    elif optionGeneration==2:
        print('Mean 1: ', xmean[0], ' Mean 2: ', xmean[1], ' Std 1: ', xmean[2], ' Std 2: ', xmean[3])
        
    outData = outdata
    outDatx = outdatx
    outSigma = outsigma
    outD=outd
    #lambdaBest = arindex[0]

    if optionGeneration==1: 
        params = xmean
        J = generate_spherical(params,numberCases,X0,Y0,X,Y,startZone,myocardium,numberPixels)
    elif optionGeneration==2:
        params = xmean
        J = generate_elliptical(params,numberCases,X0,Y0,X,Y,startZone,myocardium,numberPixels)   
    print('Counteval: ', counteval)
    return params, outData, outDatx, outSigma, outD, J, passed
