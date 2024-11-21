import numpy as np
from functions.loss import size, transmurality

def ismembertol(array,value,tol):
    maxval = np.nanmax(abs(array))
    maxval = max(maxval,value)
    answer = np.zeros(array.shape)
    answer[np.abs(array-value)<tol*maxval] = 1
    return answer

def weighted_center(realData,coords):
    coordr = coords[:,:,5,0]
    coordcirc = coords[:,:,5,1]
    mean = np.squeeze(np.nanmean(realData,axis=0))
    G = np.zeros(2)
    tmp = np.multiply(mean,coordr)
    G[0] = np.nansum(tmp)/np.nansum(mean)
    tmpX = np.multiply(mean,np.cos(coordcirc*2*np.pi))
    tmpY = np.multiply(mean,np.sin(coordcirc*2*np.pi))
    barycenter = np.zeros(2)
    barycenter[0] = np.nansum(tmpX)/np.nansum(mean)
    barycenter[1] = np.nansum(tmpY)/np.nansum(mean)
    alpha = np.arctan2(barycenter[1],barycenter[0])
    G[1] = alpha/(2*np.pi)
    return G

    
def data_preparation(realData,coords,commonMyocardium):
    G = weighted_center(realData,coords)   
    #for common myocardium
    tmp1 = np.squeeze(np.sum(realData,0))
    myocardiumR = np.logical_not(np.isnan(tmp1))
    tmp2 = np.squeeze(commonMyocardium[:,:,5])
    myocardiumS = np.logical_not(np.isnan(tmp2))
    myocardium = np.multiply(myocardiumR,myocardiumS)
    indMyocardium = np.argwhere(myocardium==1)
    p = indMyocardium.shape[0]
    nR = realData.shape[0]
    XR = np.zeros((p,nR))
    XRLatent = []    
    #to flatten infarct images
    for i in range(nR):
        tmp = realData[i,:,:]
        for j in range(p):
            XR[j,i] = tmp[indMyocardium[j,0],indMyocardium[j,1]]
        transmur = transmurality(tmp,coords)
        sz = size(tmp)
        latent = np.concatenate((transmur,np.array([sz])))
        #latent = np.array([sz])
        XRLatent.append(latent)
    XRLatent = np.array(XRLatent).T
    #to compute startZone
    rad = coords[:,:,5,0]
    circ = coords[:,:,5,1]
    I = commonMyocardium[:,:,5]
    I[I==0] = np.nan
    myocardium = I-1 #ones for myocardium
    N = I.shape[0]
    XS = np.linspace(0,N-1,N)
    X, Y = np.meshgrid(XS,XS)
    tolc = 0.025
    tolr = 0.2 
    if (G[1]-tolc<0):
        startZone = np.multiply(ismembertol(circ,G[1],tolc),ismembertol(rad,0,tolr))+np.multiply(ismembertol(1-circ,0,np.abs(G[1]-tolc)),ismembertol(rad,0,tolr))
    elif (G[1]+tolc>1):
        startZone = np.multiply(ismembertol(circ,G[1],tolc),ismembertol(rad,0,tolr))+np.multiply(ismembertol(circ,0,np.abs(1-G[1]-tolc)),ismembertol(rad,0,tolr))
    else:
        startZone = np.multiply(ismembertol(circ,G[1],tolc),ismembertol(rad,0,tolr)) 
    startZone[startZone>1] = 1
    ind = np.argwhere(startZone==1)
    Y0 = ind[:,0]
    X0 = ind[:,1]
    return G, myocardium, startZone, X0, Y0, X, Y, XR, XRLatent, indMyocardium
