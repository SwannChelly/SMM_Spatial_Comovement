np.random.seed(0)
from time import time

def g(P,sigma = 2):
    return P**(-sigma)
    
class Economy: 
    def __init__(self,R = 3, S = 1, N_upstream : np.array = None
                 , downstream : np.array = None 
                 , eta : float = 0.5, omega : np.array = None
                 , theta : float = 1, phi_bar : float = 0.9, w : np.array = None
                 ,distances : np.array = None, alpha = None , beta = None
                 ,filter_N_upstream = None,filter_A_downstream = None
                 , mu_T = 0.095,sigma_T = 1.395,sigma = 1
                 ,**kwargs):
        
        """
        R (float): number of region
        S (float): number of sectors
        N_upstream (np.array): array containing the number of firm per sector x region (s,r)
        downstream (np.array): array containing 1 when the aerospace industry has plants in this region 0 otherwise.  
        eta (float): parameter governing the elasticity of substitution for the downstream sector
        omega (np.array): array containing omega_{si}
        theta (float): shape parameter of the pareto distribution
        phi (float): scale parameter of the paretor distribution
        w (np.array): wages w_{si}
        distance (np.array): matrix of distances
        alpha: exponents for the iceberg costs according to the parametric assumption \tau_{sij} = d_{ij}^{-\alpha_s}
        beta: exponents for the search costs according to the parametric assumption \lambda_{sij} = d_{ij}^{-\beta_s}
        """
        
        ## 1) Initialisation
        t1 = time()
        self.R  = R # Number of regions
        self.S  = S # Number of sectors
        self.eta = eta # Elasticity of substitution for the aerospace production function 
        self.filter_A_downstream = filter_A_downstream
        self.sigma = sigma
        self.phi_bar = phi_bar
        self.omega = omega
        t2 = time()
        print("Init",t2-t1)
        t1 = t2

        self.distances = np.abs(np.random.random(size = (self.R,self.R))) if distances is None else distances
        if distances is None:  self.distances+= self.distances.T # Making it symmetric

        if type(alpha) == int: alpha = np.ones(self.S)*alpha
        if type(beta) == int: beta = np.ones(self.S)*beta
        
        self.tau = np.random.random(size = (self.S,self.R,self.R)) if alpha is None else distances[None, :, :] ** -alpha[:, None, None] # tau_{sij}
        self.lbd = np.random.random(size = (self.S,self.R,self.R)) if beta is None else distances[None, :, :] ** -beta[:, None, None] # lambda_{sij}
    
        self.T = np.random.normal(mu_T,sigma_T,size = (self.S,self.R))
        #self.T/=self.T[:,0].reshape(self.S,-1)
        self.T = np.exp(self.T)

        t2 = time()
        print("T",t2-t1)
        t1 = t2

        # We draw the number of firms per region and sector and collect the maximum number of firms per region
        self.N_upstream  = np.random.poisson(self.T*self.phi_bar**(-theta)) if N_upstream is None else N_upstream # N_{si} matrix
        if filter_N_upstream is not None: 
            self.N_upstream = self.N_upstream*filter_N_upstream 
        self.N = int(self.N_upstream.max())

        # We will use the upstream variable in the rest of the simulation for ease of computation. 
        # We assume that in each region there is at most N firm alive. For a region R the actual number of firms that are alive is given by self.N_upstream
        # Then we sort them for each sector on a single line in the upstream array (of size S x 1 x RN)
        self.upstream = (np.arange(self.N)[None, None, :] < self.N_upstream[:, :, None]).astype(int).reshape(self.S,1,-1)

        # Create the prices | We assume that trade frictions are similar across regions. 
        self.w  = np.abs(np.random.random(size = (S,R))) if w is None else w # Matrix of size S x R, wages
        self.w  = np.repeat(self.w.reshape(self.S,1,-1), self.N, axis=2)

        
        self.pareto_draws = np.random.pareto(theta, size=self.S*self.R*self.N).reshape(self.S,1,-1) * self.phi_bar # Pareto draw
        self.prices = self.w/self.pareto_draws
        self.prices = np.repeat(self.tau, self.N, axis=2)*self.prices 

        t2 = time()
        print("Pareto",t2-t1)
        t1 = t2

        # Create the matching
        extended_upstream = np.repeat(self.upstream,self.R,axis = 1)
        extended_lbd = np.repeat(self.lbd, self.N, axis=2)
        matching = np.random.random(size = extended_upstream.shape)>(extended_upstream*extended_lbd)
        matching = matching.astype(int)*extended_upstream

        t2 = time()
        print("Matching",t2-t1)
        t1 = t2

        # Create the network: product of prices and matching. Choose the minimum price among the suppliers set and build the network. 
        acceptable_prices = matching*(1/self.prices) # Acceptable prices contains the inverse of the prices for each supplier the downstream firms have met.
        max_indices = acceptable_prices.argmax(axis=-1)
        self.network = np.zeros_like(acceptable_prices) 
        self.network[np.arange(acceptable_prices.shape[0])[:, None], np.arange(acceptable_prices.shape[1]), max_indices] = 1

        t2 = time()
        print("Network",t2-t1)
        t1 = t2

        # Remove region where there is no plant of the aerospace industry | Can't do that otherwise there will be issue when using g. 
        # Instead use price index after creating the g values. 
        #self.network*=self.filter_A_downstream[None,:,None]

        # One can have a network through shape S x R x R x N : self.network.reshape(self.S,self.R,self.R,self.N)

        ## Create quantity and prices
        self.omega = np.random.random(size = (S,1)) if omega is None else omega # Vector of size S x 1 (same technology everywhere)
        prices_omega = (self.network*self.prices)**(1-self.eta)*self.omega[:,None,None]
        self.price_index = (prices_omega.sum(axis = -1).sum(axis = 0))**(1/(1-self.eta)) # Price index per region

        t2 = time()
        print(t2-t1)
        t1 = t2


    def trade_flow(self,X : float = 1,g = None):
        """
        Consummers have a random demand shock X_{A_j}  = X_Ag(P_j) with X_A a random demand shifter and g a function for the price index. 
        If the downstream demand is a CES, X_{A_j} = X_A P_j^{-\omega}$
        """
        # Create vector of demand for each aerospace industry 
        X_j = X*g(self.price_index,sigma = self.sigma)

        # Create trade flow. q_{sj}p_{sj} = w_{sj}X_{A_j}(p_{sj}/P_j)**(1-eta)
        # It is a matrix of size S x R x R x N
        tmp = self.omega*X_j
        trade_flows = tmp[:,:,None]*((self.prices*self.network)/self.price_index[None,:,None])**(1-self.eta)
        trade_flows = trade_flows.reshape(self.S,self.R,self.R,self.N)

        # We aggregate those at region level 
        trade_flows = trade_flows.sum(axis = -1)
        # We rearrange data such as M_{sij} is the trade flow from upstream firms in (s,i) to downstream firms in j
        trade_flows =  np.transpose(trade_flows, (0, 2, 1))
        return trade_flows
    def build_moments(self,X = 1, g = g):
        
        trade_flows = self.trade_flow(X,g)
        trade_flows = trade_flows[:,:,self.filter_A_downstream!=0]

        
        # chi_sij = M_{sij}/M_{s.j}
        chi_sij = np.zeros(shape = (self.S,self.R,self.R))
        chi_sij[:,:,self.filter_A_downstream!=0] = trade_flows/trade_flows.sum(axis = 1,keepdims = True)
           
        # chi_si = M_{si.}/M_{sA}  so far M_{sA} doesn't include flows from abroad and therefore is the total tradeflow toward aerospace industry
        chi_si = np.zeros(shape = (self.S,self.R))
        chi_si[:,self.filter_A_downstream!=0] = trade_flows.sum(axis = 1)/trade_flows.sum(axis = -1).sum(axis = -1,keepdims = True)

        # pi_sA
        pi_sA = trade_flows.sum(axis = -1).sum(axis = -1)/trade_flows.sum()

        # pi_jA
        pi_jA = np.zeros(shape = self.R)
        pi_jA[self.filter_A_downstream!=0] = trade_flows.sum(axis = 0).sum(axis = 0)/trade_flows.sum()

        moments = {"chi_sij":chi_sij,"chi_si":chi_si,"pi_sA":pi_sA,"pi_jA":pi_jA}
        return moments
        


self = Economy(**dict_var)


import torch
from time import time

def g(P, sigma=2):
    return P ** (-sigma)


class Economy:
    def __init__(self, R=3, S=1, N_upstream=None, downstream=None,
                 eta=0.5, omega=None, theta=1, phi_bar=0.9, w=None,
                 distances=None, alpha=None, beta=None,
                 filter_N_upstream=None, filter_A_downstream=None,
                 mu_T=0.095, sigma_T=1.395, sigma=1, seed=0, **kwargs):

        torch.manual_seed(seed)

        t1 = time()
        self.R = R
        self.S = S
        self.eta = eta
        self.filter_A_downstream = filter_A_downstream
        self.sigma = sigma
        self.phi_bar = phi_bar
        self.omega = omega

        t2 = time()
        print("Init", t2 - t1)
        t1 = t2

        self.distances = (torch.rand((self.R, self.R)).abs() if distances is None else distances)
        if distances is None:
            self.distances = self.distances + self.distances.T

        if isinstance(alpha, int):
            alpha = torch.ones(self.S) * alpha
        if isinstance(beta, int):
            beta = torch.ones(self.S) * beta

        self.tau = (torch.rand((self.S, self.R, self.R)) if alpha is None
                    else self.distances[None, :, :] ** -alpha[:, None, None])
        self.lbd = (torch.rand((self.S, self.R, self.R)) if beta is None
                    else self.distances[None, :, :] ** -beta[:, None, None])

        self.T = torch.exp(torch.normal(mu_T, sigma_T, size=(self.S, self.R)))

        t2 = time()
        print("T", t2 - t1)
        t1 = t2

        self.N_upstream = (torch.poisson(self.T * self.phi_bar ** (-theta)) if N_upstream is None
                           else N_upstream)
        if filter_N_upstream is not None:
            self.N_upstream = self.N_upstream * filter_N_upstream

        self.N = int(self.N_upstream.max().item())
        self.upstream = (torch.arange(self.N)[None, None, :]
                         < self.N_upstream[:, :, None]).int().reshape(self.S, 1, -1)

        self.w = (torch.rand((S, R)).abs() if w is None else w)
        self.w = self.w.reshape(self.S, 1, -1).repeat(1, 1, self.N)

        self.pareto_draws = (torch.distributions.pareto.Pareto(scale=1.0, concentration=theta)
                             .sample((self.S, 1, self.N)) * self.phi_bar)
        self.prices = self.w / self.pareto_draws
        self.prices = self.prices * self.tau.repeat(1, 1, self.N)

        t2 = time()
        print("Pareto", t2 - t1)
        t1 = t2

        extended_upstream = self.upstream.repeat(1, self.R, 1)
        extended_lbd = self.lbd.repeat(1, 1, self.N)
        matching = (torch.rand(extended_upstream.shape) > (extended_upstream * extended_lbd)).int()
        matching = matching * extended_upstream

        t2 = time()
        print("Matching", t2 - t1)
        t1 = t2

        acceptable_prices = matching * (1 / self.prices)
        max_indices = acceptable_prices.argmax(dim=-1)
        self.network = torch.zeros_like(acceptable_prices)
        arange_s = torch.arange(acceptable_prices.shape[0]).unsqueeze(1)
        arange_r = torch.arange(acceptable_prices.shape[1]).repeat(acceptable_prices.shape[0], 1)
        self.network[arange_s, arange_r, max_indices] = 1

        t2 = time()
        print("Network", t2 - t1)
        t1 = t2

        if omega is None:
            self.omega = torch.rand((S, 1))
        prices_omega = (self.network * self.prices) ** (1 - self.eta) * self.omega[:, None, None]
        self.price_index = (prices_omega.sum(-1).sum(0)) ** (1 / (1 - self.eta))

        t2 = time()
        print(t2 - t1)
        t1 = t2

    def trade_flow(self, X=1, g=g):
        X_j = X * g(self.price_index, sigma=self.sigma)
        tmp = self.omega * X_j
        trade_flows = tmp[:, :, None] * ((self.prices * self.network) / self.price_index[None, :, None]) ** (1 - self.eta)
        trade_flows = trade_flows.reshape(self.S, self.R, self.R, self.N).sum(-1)
        trade_flows = trade_flows.permute(0, 2, 1)
        return trade_flows

    def build_moments(self, X=1, g=g):
        trade_flows = self.trade_flow(X, g)
        trade_flows = trade_flows[:, :, self.filter_A_downstream != 0]

        chi_sij = torch.zeros((self.S, self.R, self.R))
        chi_sij[:, :, self.filter_A_downstream != 0] = trade_flows / trade_flows.sum(1, keepdim=True)

        chi_si = torch.zeros((self.S, self.R))
        chi_si[:, self.filter_A_downstream != 0] = trade_flows.sum(1) / trade_flows.sum((1, 2), keepdim=True).squeeze()

        pi_sA = trade_flows.sum((1, 2)) / trade_flows.sum()

        pi_jA = torch.zeros((self.R,))
        pi_jA[self.filter_A_downstream != 0] = trade_flows.sum(0).sum(0) / trade_flows.sum()

        moments = {"chi_sij": chi_sij, "chi_si": chi_si, "pi_sA": pi_sA, "pi_jA": pi_jA}
        return moments