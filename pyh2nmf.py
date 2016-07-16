import numpy as np 
from scipy.sparse.linalg import svds as svds
from sklearn.cluster import KMeans as KMeans
from scipy.sparse import spdiags as spdiags
from numpy.linalg import norm as norm
""" Simple Python Port of H2nmf"""
""" B Ravi Kiran, ENS Paris """

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
    print(M.ravel().shape)
    assert((min(M.ravel()) >= 0).all())
    
    if(algo==None):
        algo = 'h2nmf'
    # The input is a tensor --> matrix format
    if len(M.shape) == 3:
        rows,cols,bands = M.shape 
        n = rows*cols
        A = np.zeros(shape=(bands,n))
        for i in range(bands):
            A[i,:] = M[:,:,i].ravel()
        
    

    if sol == None: 
        # Intialization of the tree structure        
        sol = {}        
        sol['K'] = {}
        sol['K'][1] = range(n) # All clusters the first node contains all pixels
        sol['allnodes'] = 1 # nodes numbering
        sol['maxnode'] = 1 # Last leaf node added
        sol['parents'] = (0, 0) # Parents of leaf nodes
        sol['childs'] = {} # Child(i,:) = child of node i
        sol['leafnodes'] = [] # Current clustering: set of leaf nodes corresponding to selected clusters
        sol['leafnodes'].append(1)        
        sol['e'] = -1 # Criterion to decide which leafnode to split 
        sol['U'] = {}
        sol['U'][1] = np.ones(shape=(bands,1)) # Centroids
        sol['Ke'] = []
        sol['Ke'].append(1) # index centroid: index of the endmember
        sol['count'] = 1 # Number of clusters generated so far
        sol['firstsv'] = 0 
    

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

                    sol['parents'][sol['maxnode']+1,:] = (sol['leafnodes'][k], 0) 
                    sol['parents'][sol['maxnode']+2,:] = (sol['leafnodes'](k), 0)

                    sol['childs'][sol['leafnodes'][k], : ] = (sol['maxnode']+1, sol['maxnode']+2) 
                    sol['childs'][sol['maxnode']+1 , :] = 0 
                    sol['childs'][sol['maxnode']+2 , :] = 0 

                    sol['K'][sol['maxnode']+1] = sol['K'][sol['leafnodes'][k][Kc[1]]]  
                    sol['K'][sol['maxnode']+2] = sol['K'][sol['leafnodes'][k][Kc[2]]]

                    sol['U'][:,sol['maxnode']+1],sol['firstsv'][sol['maxnode']+1], sol['Ke'][sol['maxnode']+1] = reprvec(M[:,sol['K'][sol['maxnode']+1]]) 
                    sol['U'][:,sol['maxnode']+2],sol['firstsv'][sol['maxnode']+2], sol['Ke'][sol['maxnode']+2] = reprvec(M[:,sol['K'][sol['maxnode']+2]]) 

                    # Update criterion to choose next cluster to split
                    sol['e'][sol['maxnode']+1] = -1 
                    sol['e'][sol['maxnode']+2] = -1

                    # Compte the reduction in the error if kth cluster is split 
                    sol['e'][sol['leafnodes'][k]] = np.sqaure(sol['firstsv'][sol['maxnode']+1]) + np.sqaure(sol['firstsv'][sol['maxnode']+2]) - np.sqaure(sol['firstsv'][sol['leafnodes'][k]]) 

                    sol['maxnode'] = sol['maxnode']+2 
           


        #***************************************************************
        # Choose the cluster to split, split it, and update leaf nodes
        #***************************************************************
        if sol['count'] == 1: # Only one leaf node, the root node: split it.  
            b = 1 
        elif manualsplit == 0: # Split the node maximizing the critetion e
                [a,b] = max(sol['e'](sol['leafnodes'])) 
            
            
        sol['leafnodes'] = np.concatenate(sol['leafnodes'], sol['childs'][sol['leafnodes'][b],:]) # Add its two children
        sol['leafnodes'] = np.concatenate(sol['leafnodes'][1:b-1], sol['leafnodes'][b+1:-1]) # Remove bth leaf node

        if manualsplit == 0: # Dispay progress in tree exploration
            if mod(sol['count'],10) == 0:
                print(repr(sol['count'])) 
            if sol['count'] == r-1:
                fprintf('Done. \n',sol['count']) 
        

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
        V = np.multiply(V,np.repmat( (sum(V)+1e-16)**(-1), 2,1)) 
        x = V[0,:].T 
        # Compute treshold to split cluster 
        threshold = fquad(x) 
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
	err = np.acos( np.divide(Mm.T*u/np.norm(u)),( np.sqrt(sum(Mm**2)) ).T ) 
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
        [u,s,v] = fastsvds(M,2) 
        s1 = s(1) 
        K = FastSepNMF(s*v.T,2) 
        U = np.zeros(shape=(M.shape[0],2)) 
        if length(K) >= 1:
            U[:,0] = max(u*s*v[K[0],:].T,0) 
        
        if len(K) >= 2:
            U[:,2] = max(u*s*v[K[2],:].T,0) 
        
        # Compute corresponding optimal V 
        V = anls_entry_rank2_precompute_opt(U'*U, M'*U) 
    return (U,V,s1)
    

def FastSepNMF(M,r,normalize):

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

    if nargin <= 2: 
        normalize = 0
    if normalize == 1:
        # Normalization of the columns of M so that they sum to one
        D = spdiags((sum(M)**(-1)).T, 0, n, n) 
        M = M*D 
    

    normM = sum(M**2) 
    nM = max(normM) 

    i = 1 
    # Perform r recursion steps (unless the relative approximation error is 
    # smaller than 10^-9)
    while i <= r and max(normM)/nM > 1e-9:
        # Select the column of M with largest l2-norm
        [a,b] = max(normM) 
        # Norm of the columns of the input matrix M 
        if i == 1: 
            normM1 = normM 
        # Check ties up to 1e-6 precision
        b = np.where((a-normM)/a <= 1e-6) 
        # In case of a tie, select column with largest norm of the input matrix M 
        if length(b) > 1: 
            d = np.argmax(normM1(b)) 
            b = b(d) 
        # Update the index set, and extracted column
        J[i] = b 
        U[:,i] = M[:,b] 
        
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
        normM = normM - (v.T*M)**2 
        
        i = i + 1 
        return (J,normM,U)
        
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

    fobj = -log( np.multiply(fdel, (1-fdel) ) + exp(finter)) 
    # Can potentially use other objectives: 
    #fobj = -log( fdel.* (1-fdel) ) + 2.^(finter) 
    #fobj = ( 2*(fdel - 0.5) ).^2 + finter.^2 
    #fobj = -log( fdel.* (1-fdel) ) + finter.^2 
    #fobj = ( 2*(fdel - 0.5) ).^2 + finter.^2 
    b = np.argmin(fobj) 
    thres = delta(b) 
    return (thres,delta,fobj)

def fdelta(x,s=0.01): 
    """
    Evaluate the function fdel = sum( x_i <= delta)/n and its derivate 
    for all delta in interval [0,1] with step s
    """
    n = len(x) 
    delta = range(0,1,s) 
    lD = len(delta) 

    gs = 0.05 # Other values could be used, in [0,0.5]

    for i in range(lD):
        fdel[i] = sum(x <= delta(i))/n
        if i == 2: # use only next point to evaluate fdelp(1)
            fdelp[1] = (fdel[2]-fdel[1])/s 
        elif i >= 2: # use next and previous point to evaluate fdelp(i)
            fdelp[i-1] = (fdel[i]-fdel[i-2])/2/s 
            if i == lD: # use only previous point to evaluate fdelp(lD)
                fdelp[lD] = (fdel[lD]-fdel[lD-1])/s 
        
        deltahigh = min(1,delta(i) + gs)
        deltalow = max(0,delta(i) - gs) 
        finter[i] = ( sum(x <= deltahigh) - sum(x < deltalow) )/n/(deltahigh-deltalow) 
    return (fdel,fdelp,delta,finter,gs)
    
M = np.random.rand(100,100,400)
r = 10
algo  =1
sol = None

IDX, C, J, sol = hierclust2nmf(M,r,algo, sol)