import numpy as np 

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
             1. rank-two NMF (default)
             2. k-means
             3. spherical k-means

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
    m,n = M.shape 

    if min(M(:)) < 0
        warning('The input matrix contains negative entries which have been set to zero') 
        M = max(M,0)

    # The input is a tensor --> matrix format
    if len(M.shape) == 3
        [H,L,m] = M.shape 
        n = H*L 
        A = zeros(m,n)
        for i = 1 : m
            A(i,:) = reshape(M(:,:,i),1,n)
        end
        clear M M = A
    end
        
    if nargin == 1 || isempty(r)
        if ~exist('algo') || isempty(algo)
            algo = 1 
        end
        y = 1 n = 0 
        b = input('Do you want to visually choose the cluster to be split? (y/n) \n')
        [m,n] = size(M) 
        if b == 'y' || b == 1:
            if ~exist('L')
                H = input('What is the number of pixels in each row of your hyperspectral image? \n')
                L = n/H 
            end
            r = Inf 
            manualsplit = 1 # Split according to user feedback
        elseif b == 'n' || b == 0
            r = input('How many clusters do you want to generate? ') 
            manualsplit = 0 
        else
            error('Enter ''y'' or ''n''.')

    else: 
        manualsplit = 0 # Split according to the proposed criterion
        if nargin == 2:
            algo = 1


    if nargin < 4
        # Intialization of the tree structure
        sol.K{1} = (1:n)' # All clusters the first node contains all pixels
        sol.allnodes = 1 # nodes numbering
        sol.maxnode = 1 # Last leaf node added
        sol.parents = [0 0] # Parents of leaf nodes
        sol.childs = [] # Child(i,:) = child of node i
        sol.leafnodes = 1 # Current clustering: set of leaf nodes corresponding to selected clusters
        sol.e = -1 # Criterion to decide which leafnode to split 
        sol.U(:,1) = ones(m,1) # Centroids
        sol.Ke(1) = 1 # index centroid: index of the endmember
        sol.count = 1 # Number of clusters generated so far
        sol.firstsv = 0 
    end

    print('Hierarchical clustering started...')  

    while sol.count < r 
        
        #***************************************************************
        # Update: split leaf nodes added at previous iteration
        #***************************************************************
        for k in range(length(sol.leafnodes))
            # Update leaf nodes not yet split
            if sol.e(sol.leafnodes(k)) == -1 && length(sol.K{sol.leafnodes(k)}) > 1:
                # Update leaf node ind(k) by splitting it and adding its child nodes
                [Kc,Uc,sc] = splitclust(M(:,sol.K{sol.leafnodes(k)}),algo) 
                
                if ~isempty(Kc{2}) # the second cluster has to be non-empty: this can occur for a rank-one matrix. 
                    # Add the two leaf nodes, child of nodes(sol.leafnodes(k))
                    sol.allnodes = [sol.allnodes sol.maxnode+1 sol.maxnode+2] 

                    sol.parents(sol.maxnode+1,:) = [sol.leafnodes(k) 0] 
                    sol.parents(sol.maxnode+2,:) = [sol.leafnodes(k) 0]

                    sol.childs(sol.leafnodes(k), : ) = [sol.maxnode+1 sol.maxnode+2] 
                    sol.childs(sol.maxnode+1 , :) = 0 
                    sol.childs(sol.maxnode+2 , :) = 0 

                    sol.K{sol.maxnode+1} = sol.K{sol.leafnodes(k)}(Kc{1})  
                    sol.K{sol.maxnode+2} = sol.K{sol.leafnodes(k)}(Kc{2}) 

                    [sol.U(:,sol.maxnode+1),sol.firstsv(sol.maxnode+1), sol.Ke(sol.maxnode+1)] = reprvec(M(:,sol.K{sol.maxnode+1})) 
                    [sol.U(:,sol.maxnode+2),sol.firstsv(sol.maxnode+2), sol.Ke(sol.maxnode+2)] = reprvec(M(:,sol.K{sol.maxnode+2})) 

                    # Update criterion to choose next cluster to split
                    sol.e([sol.maxnode+1 sol.maxnode+2]) = -1 

                    # Compte the reduction in the error if kth cluster is split 
                    sol.e(sol.leafnodes(k)) = sol.firstsv(sol.maxnode+1)^2 + sol.firstsv(sol.maxnode+2)^2 - sol.firstsv(sol.leafnodes(k))^2 

                    sol.maxnode = sol.maxnode+2 
           


        #***************************************************************
        # Choose the cluster to split, split it, and update leaf nodes
        #***************************************************************
        if sol.count == 1 # Only one leaf node, the root node: split it.  
            b = 1 
        elseif manualsplit == 0 # Split the node maximizing the critetion e
                [a,b] = max(sol.e(sol.leafnodes)) 
        elseif manualsplit == 1 # Split w.r.t. user visual feedback
            [a,b] = max(sol.e(sol.leafnodes)) 
            
            close all a = affclust(sol.K(sol.leafnodes),H,L) 
            fprintf('Which cluster do you want to split (between 1 and #2.0f)? \n', sol.count)
            fprintf('Suggested cluster to split w.r.t. error: #2.0f \n', b) 
            fprintf('Type 0 if you want to stop.  \n') 
            fprintf('Type -1 if you want to fuse two clusters. \n')
            b = input('Choice: ') 
            if b == 0
                IDX = clu2vec(sol.K(sol.leafnodes))  
                for k = 1 : length(sol.leafnodes)
                    J(k) = sol.K{sol.leafnodes(k)}(sol.Ke(sol.leafnodes(k))) 
                end
                C = sol.U(:,sol.leafnodes) 
                disp('*************************************************************') 
                return 
            end
        end
        if b == -1
            fprintf('Which clusters do you want to fuse? (between 1 and #2.0f)? \n', sol.count) 
            b1 = input('Choice 1: ')
            b2 = input('Choice 2: ')
            b = sort([b1 b2]) 
            # Create a new node, child of the two fused ones, and update its entries
            sol.maxnode = sol.maxnode+1 
            sol.allnodes = [sol.allnodes sol.maxnode+1] 
            sol.parents(sol.maxnode+1,:) = [sol.leafnodes(b(1)) sol.leafnodes(b(2))] 
            sol.childs(sol.maxnode+1,:) = 0 
            sol.K{sol.maxnode+1} = [sol.K{sol.leafnodes(b(1))} sol.K{sol.leafnodes(b(2))}]  
            [u1,s1,ke1] = reprvec(M(:,sol.K{sol.maxnode+1})) 
            sol.firstsv(sol.maxnode+1) = s1
            sol.U(:,sol.maxnode+1) = u1 
            sol.Ke(:,sol.maxnode+1) = ke1 
            sol.e([sol.maxnode+1]) = -1 
            sol.e2([sol.maxnode+1]) = -1
            
            # Update leaf nodes: delete two fused and add the new one
            sol.leafnodes =  sol.leafnodes([1:b(1)-1 b(1)+1:b(2)-1  b(2)+1:end])
            sol.leafnodes = [sol.leafnodes sol.maxnode+1] 
                 
            # Update counters
            sol.maxnode = sol.maxnode+1 
            sol.count = sol.count-1 
        else:
            
            sol.leafnodes = [sol.leafnodes sol.childs(sol.leafnodes(b),:)'] # Add its two children
            sol.leafnodes = sol.leafnodes([1:b-1 b+1:end]) # Remove bth leaf node

            if manualsplit == 0: # Dispay progress in tree exploration
                if mod(sol.count,10) == 0:
                    fprintf('#1.0f...\n',sol.count) 
                else:
                    fprintf('#1.0f...',sol.count) 
                end
                if sol.count == r-1:
                    fprintf('Done. \n',sol.count) 
                
            else
                disp('*************************************************************') 
            

            sol.count = sol.count+1 
    
    IDX = clu2vec(sol.K(sol.leafnodes))  
    for k = 1 : length(sol.leafnodes)
        J(k) = sol.K{sol.leafnodes(k)}(sol.Ke(sol.leafnodes(k))) 
    
    C = sol.U(:,sol.leafnodes) 

    return (IDX, C, J, sol)

def splitclust(M,algo): 
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
    if nargin == 1:
        algo = 1:
    
    if algo == 1  # rank-2 NMF
        [U,V,s] = rank2nmf(M): 
        # Normalize columns of V to sum to one
        V = V.*repmat( (sum(V)+1e-16).^(-1), 2,1): 
        x = V(1,:)': 
        # Compute treshold to split cluster 
        threshold = fquad(x): 
        K{1} = find(x >= threshold): 
        K{2} = find(x < threshold):  
        
    elif algo == 2: # k-means
        [u,s,v] = fastsvds(M,2): # Initialization: SVD+SPA
        Kf = FastSepNMF(s*v',2,0):
        U0 = u*s*v(Kf,:)': 

        [IDX,U] = kmeans(M', 2, 'EmptyAction','singleton','Start',U0'): 
        U = U': 
        K{1} = find(IDX==1): 
        K{2} = find(IDX==2): 
        s = s(1): 
        
    elif algo == 3: # shperical k-means
        [u,s,v] = fastsvds(M,2): # Initialization: SVD+SPA 
        Kf = FastSepNMF(s*v',2,0):
        U0 = u*s*v(Kf,:)': 
        
        [IDX,U] = spkmeans(M, U0): 
        # or (?)
        #[IDX,U] = kmeans(M', 2, 'EmptyAction','singleton','Start',U0','Distance','cosine'): 
        K{1} = find(IDX==1): 
        K{2} = find(IDX==2): 
        s = s(1): 
    
    return (K,U,s)