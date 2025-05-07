import numpy as np
import scipy
import math
from sklearn.manifold import Isomap
from functions.generations import generate_elliptical, generate_spherical

def size(image2d):
    size = np.sum(image2d>1.5)/np.sum(image2d>0)  
    return size

def transmurality(image2d,coords):
    coordcirc = coords[:,:,5,1]
    uniqueangles = np.linspace(0,1,25)
    transmurality = np.zeros(uniqueangles.shape[0]-1)
    for s in range(uniqueangles.shape[0]-1):
        countinfarct = 0
        countnoninfarct = 0
        cond1 = uniqueangles[s]<=coordcirc
        cond2 = coordcirc<uniqueangles[s+1]
        cond = np.multiply(cond1,cond2)
        indlist = np.argwhere(cond)
        for k in range(indlist.shape[0]):
            if image2d[indlist[k][0],indlist[k][1]]>1.5:
                countinfarct = countinfarct+1
            elif image2d[indlist[k][0],indlist[k][1]]<=1.5:
                countnoninfarct = countnoninfarct+1         
        transmurality[s]=countinfarct/(countinfarct+countnoninfarct)
    return transmurality 
    
def kde(XR,XS,nR,kNN):
    X = np.concatenate((XR, XS), axis=1)
    tmpKS = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(X.T))
    tmp = tmpKS + np.diag(math.inf*np.ones(X.shape[1]))
    tmpB = np.sort(tmp,axis=0)
    tmpB = tmpB[kNN,:]
    sigma = np.mean(tmpB)
    K = np.exp(-tmpKS**2 / (2*sigma**2))
    pdfXR = np.sum(K[0:nR,:],axis=0)
    pdfXR = pdfXR/(nR*sigma*np.sqrt(2*np.pi))
    pdfXR = pdfXR/np.sum(pdfXR)
    pdfXS = np.sum(K[nR:,:],axis=0)
    pdfXS = pdfXS/((X.shape[1]-nR)*sigma*np.sqrt(2*np.pi))
    pdfXS = pdfXS/np.sum(pdfXS)

    return pdfXR, pdfXS
    


def loss_function(optionGeneration,optionLoss,params,numberCases,XR,XRLatent,indMyocardium,knn,X0,Y0,X,Y,startZone,myocardium,numberPixels,coords):
    if optionGeneration==1: #spherical
        J = generate_spherical(params,numberCases,X0,Y0,X,Y,startZone,myocardium,numberPixels)
    elif optionGeneration==2: #elliptical
        J = generate_elliptical(params,numberCases,X0,Y0,X,Y,startZone,myocardium,numberPixels)
    if optionLoss==2:
        nR = XRLatent.shape[1]
        XSLatent = []
        nS = numberCases
        for i in range(nS):
            tmp = J[:,:,i]
            transmur = transmurality(tmp,coords)
            sz = size(tmp)
            latent = np.concatenate((transmur,np.array([sz])))
            XSLatent.append(latent)
        XSLatent = np.array(XSLatent).T
    else:
        nR = XR.shape[1]
        nS = numberCases
        p = indMyocardium.shape[0]
        XS = np.zeros((p,nS))
        for i in range(nS):
            tmp = J[:,:,i]
            for k in range(p):
                XS[k,i] = tmp[indMyocardium[k,0],indMyocardium[k,1]]                
    if optionLoss==1:
        pdfXR, pdfXS = kde(XR,XS,nR,knn)
        KLtmp = np.multiply(pdfXS,np.log(np.divide(pdfXS,pdfXR)))
        L = np.sum(KLtmp) 
    elif optionLoss==2:
        pdfXR, pdfXS = kde(XRLatent,XSLatent,nR,knn)
        KLtmp = np.multiply(pdfXS,np.log(np.divide(pdfXS,pdfXR)))
        L = np.sum(KLtmp) 
    return L

