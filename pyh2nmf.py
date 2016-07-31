import numpy as np 
from scipy.sparse.linalg import svds as svds
from sklearn.cluster import KMeans as KMeans
from scipy.sparse import spdiags as spdiags
from numpy.linalg import norm as norm
from numpy.matlib import repmat as repmat
from scipy.io import loadmat as loadmat
from numpy.matlib import repmat
from scipy import sparse

""" Simple Python Port of H2nmf by Gillis et al. 2014"""
""" B Ravi Kiran, ENS Paris June 2016 """

def hierclust2nmf(M,r,algo,sol):
    """
    Hierarchical custering based on rank-two nonnegative matrix factorization
    Given a data matrix M (m-by-n) representing n data points in an
    m-dimensional space, this algorithm computes a set of clusters obtained
    using the hierarchical rank-two NMF method described in  
    Gillis, Kuang, Park, `Hierarchical Clustering of Hyperspectral Images 
    using Rank-Two Nonnegative Matrix Factorization', arXiv. 

    ****** Input ******
     M     : m-by-n data matrix (or a H-by-L-by-m tensor)
             n (=HL) is the number of pixels, m the number of wavelengths          
     r     : number of clusters to generate, OR 
             for r = []:  the user is asked how many clusters she/he wants, 
                          OR she/he can choose the leaf node to be split based 
                          on the displayed current clusters. (default)
     algo  : algorithm used to split the clusters
             1. h2nmf  :rank-two NMF (default)
             2. hkm : k-means
             3. hskm : spherical k-means

     sol : it can be used to resume a previous solution computed by 
           hierclust2nmf (this allows to go deeper in the tree and 
           generate more clusters from a previous solution). 
           ---> Optional.
    ****** Output ******
      IDX  : an indicator vector in {1,2,...,r}^m, identifying r clusters
      C, J : The columns of C are the spectral signatures of endmembers, 
             that is, the cluster centroids. 
             J is the index set locating the endmembers in the data set:
             C = M(:,J) (for matrix format). 
      sol  : contains the tree structure
    """
    
    # The input matrix contains negative entries which have been set to zero    
    assert((min(M.ravel()) >= 0).all())
    manualsplit = 0
    
    if(algo==None):
        algo = 1
    
    # The input is a tensor --> matrix format
    if len(M.shape) == 3:
        rows,cols,bands = M.shape 
        n = rows*cols
        #fortran ordering reshape
        M = M.reshape((rows*cols,bands), order ='F').transpose()
    

    if sol == None: 
        # Intialization of the tree structure        
        sol = {}        
        sol['K'] = {}
        sol['K'][1] = np.array(range(n),dtype=int) # All clusters the first node contains all pixels
        sol['allnodes'] = []        
        sol['allnodes'].append(1) # nodes numbering
        sol['maxnode'] = 1 # Last leaf node added
        sol['parents'] = {}
        sol['parents'][1] = (0, 0) # Parents of leaf nodes
        sol['childs'] = {} # Child(i,:) = child of node i
        sol['leafnodes'] = [] # Current clustering: set of leaf nodes corresponding to selected clusters
        sol['leafnodes'].append(1)        
        sol['e'] = {}        
        sol['e'][1] = -1 # Criterion to decide which leafnode to split 
        sol['U'] = {}
        sol['U'][1] = np.ones(shape=(bands)) # Centroids
        sol['Ke'] = {}
        sol['Ke'][sol['maxnode']] = 1 # index centroid: index of the endmember
        sol['count'] = 1 # Number of clusters generated so far
        sol['firstsv'] = {}
        sol['firstsv'][sol['maxnode']] = 0
    

    print('Hierarchical clustering started...')  

    while sol['count'] < r:
        
        #***************************************************************
        # Update: split leaf nodes added at previous iteration
        #***************************************************************
        for k in range(len(sol['leafnodes'])):            
            # Update leaf nodes not yet split
            if sol['e'][sol['leafnodes'][k]] == -1 and len(sol['K'][sol['leafnodes'][k]]) > 1:
                # Update leaf node ind(k) by splitting it and adding its child nodes
                Kc,Uc,sc = splitclust(M[:,sol['K'][sol['leafnodes'][k]]],algo) 
                
                if Kc[2] : # the second cluster has to be non-empty: this can occur for a rank-one matrix. 
                    # Add the two leaf nodes, child of nodes(sol['leafnodes'](k))
                    sol['allnodes'] = sol['allnodes'].append((sol['maxnode']+1,sol['maxnode']+2)) 

                    sol['parents'][sol['maxnode']+1] = (sol['leafnodes'][k], 0) 
                    sol['parents'][sol['maxnode']+2] = (sol['leafnodes'][k], 0)

                    sol['childs'][sol['leafnodes'][k]] = (sol['maxnode']+1, sol['maxnode']+2) 
                    sol['childs'][sol['maxnode']+1] = 0 
                    sol['childs'][sol['maxnode']+2] = 0 

                    sol['K'][sol['maxnode']+1] = sol['K'][sol['leafnodes'][k]][Kc[1]]  
                    sol['K'][sol['maxnode']+2] = sol['K'][sol['leafnodes'][k]][Kc[2]]

                    sol['U'][sol['maxnode']+1], sol['firstsv'][sol['maxnode']+1], sol['Ke'][sol['maxnode']+1] = reprvec(M[:,sol['K'][sol['maxnode']+1]]) 
                    sol['U'][sol['maxnode']+2], sol['firstsv'][sol['maxnode']+2], sol['Ke'][sol['maxnode']+2] = reprvec(M[:,sol['K'][sol['maxnode']+2]]) 

                    # Update criterion to choose next cluster to split
                    sol['e'][sol['maxnode']+1] = -1 
                    sol['e'][sol['maxnode']+2] = -1

                    # Compte the reduction in the error if kth cluster is split 
                    sol['e'][sol['leafnodes'][k]] = (sol['firstsv'][sol['maxnode']+1])**2 + (sol['firstsv'][sol['maxnode']+2])**2 - (sol['firstsv'][sol['leafnodes'][k]])**2 

                    sol['maxnode'] = sol['maxnode']+2 
           


        #***************************************************************
        # Choose the cluster to split, split it, and update leaf nodes
        #***************************************************************
        if sol['count'] == 1: # Only one leaf node, the root node: split it.  
            b = 0 
        elif manualsplit == 0: # Split the node maximizing the critetion e
            b = np.argmax(sol['e'](sol['leafnodes'])) 
            
        #update leaves
        sol['leafnodes'].append(sol['childs'][sol['leafnodes'][b]]) # Add its two children
        sol['leafnodes'] = np.concatenate(sol['leafnodes'][0:b-1], sol['leafnodes'][b+1:-1]) # Remove bth leaf node

        sol['count'] = sol['count']+1 
    
    IDX = clu2vec(sol['K'](sol['leafnodes']))  
    #dictionary of indices    
    J = {}    
    for k in range(len(sol['leafnodes'])):
        J[k] = sol['K'][sol['leafnodes'][k]][sol['Ke'][sol['leafnodes'][k]]] 
    
    C = sol['U'][:,sol['leafnodes']] 

    return (IDX, C, J, sol)

def splitclust(M,algo=1): 
    """
    Given a matrix M, split its columns into two subsets

    See Section 3 in 
    Gillis, Kuang, Park, `Hierarchical Clustering of Hyperspectral Images 
    using Rank-Two Nonnegative Matrix Factorization', arXiv. 
    ****** Input ******
     M     : m-by-n data matrix (or a H-by-L-by-m tensor) 
     algo  : algorithm used to split the clusters
             1. rank-two NMF (default)
             2. k-means
             3. spherical k-means
    ****** Output ******
      K    : two clusters 
      U    : corresponding centroids
      s    : first singular value of M
    """
    
    if algo == 1:  # rank-2 NMF
        U,V,s = rank2nmf(M)
        # Normalize columns of V to sum to one
        V = np.multiply(V,repmat( (sum(V)+1e-16)**(-1), 2,1)) 
        x = V[0,:].T 
        # Compute treshold to split cluster 
        threshold,_,_ = fquad(x) 
        K = {} #children dictionary
        K[1] = np.where(x >= threshold) 
        K[2] = np.where(x < threshold) 
        
    elif algo == 2: # k-means
        [u,s,v] = fastsvds(M,2) # Initialization: SVD+SPA
        Kf = FastSepNMF(s*v.T,2,0)
        U0 = u*s*v[Kf,:].T 

        IDX,U = KMeans(M.T, 2, 'EmptyAction','singleton','Start',U0.T)
        U = U.T 
        K = {} #children dictionary
        K[1] = np.where(IDX==1) 
        K[2] = np.where(IDX==2)
        s = s[1]
        
    elif algo == 3: # shperical k-means
        u,s,v = fastsvds(M,2) # Initialization: SVD+SPA 
        Kf = FastSepNMF(s*v.T,2,0)
        U0 = u*s*v[Kf,:].T 
        
        IDX,U = spkmeans(M, U0) 
        # or (?)
        #[IDX,U] = kmeans(M', 2, 'EmptyAction','singleton','Start',U0','Distance','cosine'): 
        K[1] = np.where(IDX==1) 
        K[2] = np.where(IDX==2) 
        s = s[1]
    
    return (K,U,s)


def clu2vec(K,m,r): 
    
    """ Transform a cluster cell to a vector """ 
    if r == none:
        r = len(K)

    if m == none: # Compute max entry in K
        m = 0 
        for i in range(r):
            m = max(0, max(K[i])) 
	    
    IDX = np.zeros(shape=(m,1)) 

    for i in range(r):
        IDX[K[i]] = i 
        
    return IDX

def reprvec(M):
	"""
	 Extract "most" representative column from a matrix M as follows: 
	 
	 First, it computes the best rank-one approximation u v^T of M. 
	 Then, it identifies the column of M minimizing the MRSA with the first
	 singular vector u of M. 
	 
	 See Section 4.4.1 of 
	 Gillis, Kuang, Park, `Hierarchical Clustering of Hyperspectral Images 
	 using Rank-Two Nonnegative Matrix Factorization', arXiv. 
	"""

	u,s,v = svds(M,1) 
	u = np.abs(u) 
	m,n = M.shape 
	#Exctract the column of M approximating u the best (up to a translation and scaling)
	u = u - np.mean(u) 
	Mm = M - np.mean(M)  
	err = np.arccos( np.divide(np.dot(Mm.T,u/norm(u)),(np.sqrt(np.sum(Mm**2))).T) ) 
	b = np.argmin(err) 
	u = M[:,b] 
	return u,s,b
    

def rank2nmf(M):
    """
     Given a data matrix M (m-by-n), computes a rank-two NMF of M. 

     See Algorithm 3 in 
     
     Gillis, Kuang, Park, `Hierarchical Clustering of Hyperspectral Images 
     using Rank-Two Nonnegative Matrix Factorization', arXiv.  
     
     ****** Input ******
      M     : a nonnegative m-by-n data matrix  

     ****** Output ******
      (U,V) : a rank-two NMF of M
      s1    : first singular value of M 
    """

    m,n = M.shape

    # Best rank-two approximation of M
    if min(m,n) == 1:
        U,S,V = fastsvds(M,1) 
        U = np.abs(U) 
        V = np.abs(V) 
        s1 = S 
    else:
        u,s,v = fastsvds(M,2) 
        s1 = s[0]
#        print(v.shape, s.shape, np.dot(s,v.T).shape)
        K,_,_ = FastSepNMF(np.dot(s,v.T),2) 
        
        U = np.zeros(shape=(M.shape[0],2)) 
        if len(K) >= 1:
            us = np.dot(u,s)
            U[:,0] = np.maximum(np.dot(us,v[K[0],:]),0) 
        
        if len(K) >= 2:
            us = np.dot(u,s)
            U[:,1] = np.maximum(np.dot(us,v[K[1],:]),0) 
        
        # Compute corresponding optimal V 
        V = anls_entry_rank2_precompute_opt(np.dot(U.T,U), np.dot(M.T,U)) 
    return (U,V,s1)
    

def fastsvds(M,r): 
    """
    "Fast" but less accurate SVD by computing the SVD of MM^T or M^TM 
    ***IF*** one of the dimensions of M is much smaller than the other. 
    Note. This is numerically less stable, but useful for large hyperspectral 
    images. 

    """

    m,n = M.shape 
    rationmn = 10 # Parameter, should be >= 1

    if m < rationmn*n: 
        MMt = np.dot(M,M.T)
        u,s,v = svds(MMt,r)
        s = np.diag(s)
        v = np.dot(M.T, u) 
        v = np.multiply(v,repmat( (sum(v**2)+1e-16)**(-0.5),n,1)) 
        s = np.sqrt(s) 
    elif n < rationmn*m:
        MtM = np.dot(M.T,M)
        u,s,v = svds(MtM,r) 
        s = np.diag(s)
        u = np.dot(M,v) 
        u = np.multiply(u,repmat( (sum(u**2)+1e-16)**(-0.5),m,1))
        s = np.sqrt(s) 
    else:
        u,s,v = svds(M,r) 
        s = np.diag(s)
    return (u,s,v)
    
def FastSepNMF(M,r,normalize=0):

    """
     FastSepNMF - Fast and robust recursive algorithm for separable NMF
     
     *** Description ***
     At each step of the algorithm, the column of M maximizing ||.||_2 is 
     extracted, and M is updated by projecting its columns onto the orthogonal 
     complement of the extracted column. 

     See N. Gillis and S.A. Vavasis, Fast and Robust Recursive Algorithms 
     for Separable Nonnegative Matrix Factorization, arXiv:1208.1237. 
     
     See also https://sites.google.com/site/nicolasgillis/

     [J,normM,U] = FastSepNMF(M,r,normalize) 

     ****** Input ******
     M = WH + N : a (normalized) noisy separable matrix, that is, W is full rank, 
                  H = [I,H']P where I is the identity matrix, H'>= 0 and its 
                  columns sum to at most one, P is a permutation matrix, and
                  N is sufficiently small. 
     r          : number of columns to be extracted. 
     normalize  : normalize=1 will scale the columns of M so that they sum to one,
                  hence matrix H will satisfy the assumption above for any
                  nonnegative separable matrix M. 
                  normalize=0 is the default value for which no scaling is
                  performed. For example, in hyperspectral imaging, this 
                  assumption is already satisfied and normalization is not
                  necessary. 

     ****** Output ******
     J        : index set of the extracted columns. 
     normM    : the l2-norm of the columns of the last residual matrix. 
     U        : normalized extracted columns of the residual. 

     --> normM and U can be used to continue the recursion later on without 
         recomputing everything from scratch. 

     This implementation of the algorithm is based on the formula 
     ||(I-uu^T)v||^2 = ||v||^2 - (u^T v)^2. 
    """
    m,n = M.shape
    J = []
    U = np.zeros((m,r)) #this is the maximal size
    if normalize == 1:
        # Normalization of the columns of M so that they sum to one
        D = spdiags((sum(M)**(-1)).T, 0, n, n) 
        M = np.dot(M,D) 
    

    normM = np.sum(M**2,axis=0) 
    nM = np.max(normM) 
    #python indexing
    i = 0 
    # Perform r recursion steps (unless the relative approximation error is 
    # smaller than 10^-9)
    while i < r and np.max(normM)/nM > 1e-9:
        # Select the column of M with largest l2-norm
        a = np.amax(normM)        
        #b = np.argmax(normM) 
        # Norm of the columns of the input matrix M 
        if i == 1: 
            normM1 = normM 
        # Check ties up to 1e-6 precision
        b = np.where((a-normM)/a <= 1e-6) 
        # In case of a tie, select column with largest norm of the input matrix M 
#        print(b)
        if len(b) > 1: 
            d = np.argmax(normM1[b]) 
            b = b[d] 
        
        assert(len(b)==1)
        # Update the index set, and extracted column
        J.append(b)
        U[:,i] = M[:,b].ravel()
        
        # Compute (I-u_{i-1}u_{i-1}^T)...(I-u_1u_1^T) U(:,i), that is, 
        # R^(i)(:,J(i)), where R^(i) is the ith residual (with R^(1) = M).
        for j in range(i-2):
            U[:,i] = U[:,i] - U[:,j]*(U[:,j].T*U[:,i])
        
        # Normalize U[:,i]
        U[:,i] = U[:,i]/norm(U[:,i]) 
        
        # Compute v = u_i^T(I-u_{i-1}u_{i-1}^T)...(I-u_1u_1^T)
        v = U[:,i] 
        for j in range(i-1,1,-1):
            v = v - (v.T*U[:,j])*U[:,j] 
        
        
        # Update the norm of the columns of M after orhogonal projection using
        # the formula ||r^(i)_k||^2 = ||r^(i-1)_k||^2 - ( v^T m_k )^2 for all k. 
        normM = normM - (np.dot(v.T,M))**2 
        
        i = i + 1 
    return (np.squeeze(J), normM, U)
        
def fquad(x,s=0.01): 
    """
    Select treshold to split the entries of x into two subsets
    See Section 3.2 in 
    Gillis, Kuang, Park, `Hierarchical Clustering of Hyperspectral Images 
    using Rank-Two Nonnegative Matrix Factorization', arXiv. 
    """

    fdel,fdelp,delta,finter,gs = fdelta(x,s) 
    # fdel is the percentage of values smaller than delta
    # finter is the number of points in a small interval around delta

    fobj = -np.log( np.multiply(fdel, (1-fdel) ) + np.exp(finter)) 
    # Can potentially use other objectives: 
    #fobj = -log( fdel.* (1-fdel) ) + 2.^(finter) 
    #fobj = ( 2*(fdel - 0.5) ).^2 + finter.^2 
    #fobj = -log( fdel.* (1-fdel) ) + finter.^2 
    #fobj = ( 2*(fdel - 0.5) ).^2 + finter.^2 
    b = np.argmin(fobj) 
    thres = delta[b] 
    return (thres,delta,fobj)

def fdelta(x,s=0.01): 
    """
    Evaluate the function fdel = sum( x_i <= delta)/n and its derivate 
    for all delta in interval [0,1] with step s
    """
    n = len(x) 
    delta = np.arange(0,1,s) 
    lD = len(delta) 

    gs = 0.05 # Other values could be used, in [0,0.5]
    fdel = np.zeros((lD))
    finter = np.zeros((lD))
    fdelp = np.zeros((lD))
    for i in range(lD):
        fdel[i] = sum(x <= delta[i])/n
        if i == 1: # use only next point to evaluate fdelp(1)
            fdelp[0] = (fdel[1]-fdel[0])/s 
        elif i >= 1: # use next and previous point to evaluate fdelp(i)
            fdelp[i-1] = (fdel[i]-fdel[i-2])/2/s 
            if i == lD: # use only previous point to evaluate fdelp(lD)
                fdelp[lD] = (fdel[lD]-fdel[lD-1])/s 
        
        deltahigh = np.minimum(1,delta[i] + gs)
        deltalow = np.maximum(0,delta[i] - gs) 
        finter[i] = ( sum(x <= deltahigh) - sum(x < deltalow) )/n/(deltahigh-deltalow) 
    return (fdel,fdelp,delta,finter,gs)
    
def anls_entry_rank2_precompute_opt(left, right):
    """
    Solve min_H ||M - WH'||_2 s.t. H >= 0

    where left = W^TW and right = M^TW 

    See Kuang, Park, `Fast Rank-2 Nonnegative Matrix Factorization 
    for Hierarchical Document Clustering', KDD '13. 

    See also Algorithm 4 in 
    Gillis, Kuang, Park, `Hierarchical Clustering of Hyperspectral Images 
    using Rank-Two Nonnegative Matrix Factorization', arXiv. 

    ****** Input ******
    left     : 2-by-2 matrix (or possibly 1-by-1)
    right    : n-by-2 matrix (or possibly n-by-1)

    ****** Output ******
    H    : nonnegative n-by-2 matrix, solution to KKT equations
    """

    if len(left) == 1:
        H = max(0,right/left)
    else:
        H = np.linalg.solve(left, right.T).T 
        #need to find a dimension wise .all() flag (this does not work for now)
        #original ~all(H>=0, 2)
        use_either = np.all(H>=0,axis=1)
        H[use_either, :] = anls_entry_rank2_binary(left, right[use_either,:]) 
        H = H.T 

    return H


def anls_entry_rank2_binary(left, right):
    """ Case where one entry in each column of H has to be equal to zero """
    n = right.shape[0]

    solve_either = np.zeros(shape=(n, 2))
    solve_either[:, 0] = np.maximum(0, np.divide(right[:, 0], left[0,0])) 
    solve_either[:, 1] = np.maximum(0, np.divide(right[:, 1], left[1,1])) 

    cosine_either = np.multiply(solve_either, repmat([np.sqrt(left[0,0]), np.sqrt(left[1,1])],n,1) )

    choose_first = (cosine_either[:, 0] >= cosine_either[:, 1])
    solve_either[choose_first, 1] = 0
    solve_either[~choose_first, 0] = 0
    return solve_either

def spkmeans(X, init):
        
    """
    Perform spherical k-means clustering.
    X: d x n data matrix
    init: k (1 x 1) or label (1 x n, 1<=label(i)<=k) or center (d x k)
    Reference: Clustering on the Unit Hypersphere using Von Mises-Fisher Distributions.
    by A. Banerjee, I. Dhillon, J. Ghosh and S. Sra.
    Written by Michael Chen (sth4nth@gmail.com).
    Based on matlab version @ 
    http://www.mathworks.com/matlabcentral/fileexchange/28902-spherical-k-means/content/spkmeans.m 
    (and slightly modifed to run on previous verions of Matlab)
    initialization
    """
    d,n = X.shape

    if n <= init:
        label = range(1,init) 
        m = X 
        energy = 0
    else:
        # Normalize the columns of X
        X = np.dot(X, repmat( (sum(X**2)+1e-16)**(-0.5),d,1)) 

        if len(init) == 1:
            idx = randsample(n,init)
            m = X[:,idx]
            [ul,label] = np.maximum(np.dot(m.T,X),[],1)
        elif init.shape[0] == 1 and init.shape[1] == n:
            label = init
        elif init.shape[0] == d:
            m = np.multiply(init, repmat( (sum(init**2)+1e-16)**(-0.5),d,1))
            ul,label = np.maximum(np.dot(m.T,X),[],1)
        else:
            error('ERROR: init is not valid.')
		
		## main algorithm: final version 
        last = 0
        while (label != last).any():
            u,pipi,label = np.unique(label)   # remove empty clusters
            k = len(u)
            E = sparse.coo_matrix(range(n),label,1,n,k,n)
            m = np.dot(X,E) 
            m = np.dot(m, repmat( (sum(m**2)+1e-16)**(-0.5),d,1)) 
            last = label
            val = np.maximum(np.dot(m.T*X),[],1)
            label = np.argmax(np.dot(m.T*X),[],1)
        ul,ul,label = np.unique(label)   # remove empty clusters
        energy = np.sum(val)
    return (label, m, energy)
    
matlab_mat = loadmat('Urban.mat')
#pick every 10x10 patch with 20 lambdas (faster for test)
M = matlab_mat['R'][:10,:10,:20].astype(float)
#r-clusters
r = 5
#1 h2nmf, 2 hkm, 3 hskm
algo = 1
#no initial tree structure
sol = None

IDX, C, J, sol = hierclust2nmf(M,r,algo, sol)