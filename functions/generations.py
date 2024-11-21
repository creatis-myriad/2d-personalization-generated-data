import numpy as np


def cart2pol(x, y):
    rho = np.sqrt(x**2+y**2)
    phi = np.arctan2(y,x)
    return(rho, phi)
    
    
def generate_elliptical(params,numberCases,X0,Y0,X,Y,startZone,myocardium,numberPixels):
    amean = params[0]
    bmean = params[1]
    astd = params[2]
    bstd = params[3]  
    J = np.zeros((numberPixels,numberPixels,numberCases))   
    for k in range(numberCases):
        I = np.copy(myocardium)
        infarctZone = np.zeros(I.shape)
        r = np.random.randint(0,Y0.shape[0],size=1)        
        x0 = X0[r]
        y0 = Y0[r]  
        a = np.random.normal(amean,astd,1)
        b = np.random.normal(bmean,bstd,1)
        R, TH = cart2pol(X-x0,Y-y0)
        Rorigine, THorigine = cart2pol(x0-numberPixels//2,y0-numberPixels//2)
        TH = TH-THorigine
        target = ((R*np.cos(TH))**2)/b**2+((R*np.sin(TH))**2)/a**2<1
        infarctZone[target] = 1        
        I = I+1*infarctZone
        I[I>2] = 2
        J[:,:,k] = I
    return J
    
    
def generate_spherical(params,numberCases,X0,Y0,X,Y,startZone,myocardium,numberPixels):
    numIt = int(params[0])
    maxRad = params[1]
    J = np.zeros((numberPixels,numberPixels,numberCases))    
    for k in range(numberCases):
        I = np.copy(myocardium)
        infarctZone = np.zeros(I.shape)       
        for it in range(numIt):  
            if it==0:
                tmpI = startZone
                R = np.random.rand(1,1)*(maxRad-1)+1           
            targetZone = np.argwhere(tmpI==1)
            cid = np.random.randint(0,targetZone.shape[0],size=1)
            y = targetZone[cid,0]
            x = targetZone[cid,1]            
            target = (X-x)**2+(Y-y)**2<R**2
            tmpI = np.zeros(I.shape)
            tmpI[target] = 1
            tmpI = np.multiply(tmpI,myocardium)
            infarctZone[target] = 1        
            I = I+1*infarctZone
            I[I>2] = 2
            R = np.random.rand(1,1)*(maxRad-1)+1 
        J[:,:,k] = I
    return J