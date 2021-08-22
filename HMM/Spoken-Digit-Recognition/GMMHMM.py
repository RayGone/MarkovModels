from logging import raiseExceptions
from scipy import stats as st
import numpy as np

class GHMM:
    def __init__(self,n_states) -> None:
        self.ns = n_states # number of hidden states

        self.A = self._stochastize(np.random.RandomState(0).rand(self.ns,self.ns)) #np.random.randint(0,100)).rand(self.n_states,self.n_states))
        self.pi = self._normalize(np.random.RandomState(0).rand(self.ns,1))

        self.mu = None # mean vectors - of states
        self.cov = None # covariance matrices - of states
        self.nd = None # dimension of input - should be equal to number of hidden states i.e. self.ns
    
    def _normalize(self,X):
        X = (X + (X==0))/np.sum(X)
        return X

    def _stochastize(self,Y):
        Y = (Y+(Y==0))/np.sum(Y,axis=1)
        return Y

    def _initialize_state_gaussian(self,obs):
        '''
            # Initializes mean vector and covariance matrix for the states.
            Each state is inialized to be sort of random gaussian, based on the first training example. 
            Each state_gaussian is initialized to have same covariance which is in fact the covariance of first training example
        '''
        if(self.nd == None):
            self.nd = obs.shape[0]
        if(self.mu == None):
            subset = np.random.RandomState(1).choice(np.arange(self.nd),self.ns,replace=False)
            self.mu = obs[:,subset]
        if(self.cov == None):
            self.cov = np.zeros((self.ns,self.nd,self.nd))
            self.cov += np.cov(obs)
            # print(self.cov)

    def _calculate_emission_probability(self,obs):
        '''
        # Calculates Emission Probability; B matrix.
        i.e The probability of ith observation being generated by jth state
        '''
        obs = np.atleast_2d(obs)
        self.B = np.zeros((self.ns,obs.shape[1]))
        for s in range(self.ns):
            self.B[s,:] = st.multivariate_normal.pdf(obs.T,mean=self.mu[s],cov=self.cov[s])

    def _forward(self):
        '''
        # Forward Algorithm
        first part of forward-backward Algorithm
        '''
        self.alpha = np.zeros(self.B.shape)
        log_likelihood = 0.0
        T = self.B.shape[1]
        for t in range(T):
            if t == 0:
                self.alpha[:,t] = self.B[:,t] * self.pi.ravel()
            else:
                self.alpha[:,t] = self.B[:,t] * np.dot(self.A.T,self.alpha[:,t-1])
        
            alpha_sum = np.sum(self.alpha[:,t])
            self.alpha[:,t] /= alpha_sum
            log_likelihood = log_likelihood + np.log(alpha_sum)
        return log_likelihood

    def _backward(self):
        '''
        # Backward Algorithm
        second part of forward-backward algorithm
        '''
        self.beta = np.ones(self.B.shape)
        T = self.B.shape[1]        
        self.beta[:, -1] = np.ones(self.B.shape[0])

        for t in range(T-1)[::-1]:
            self.beta[:,t] = np.dot(self.A, self.B[:,t+1]*self.beta[:,t+1])
            self.beta[:,t] /= np.sum(self.beta[:,t])

    def _Baum_Welch_Algorithm(self,obs):
        '''
        # Baum-Welch Algorithm
        This is EM algorithm that uses forward-backward algorithm.
        '''
        T = obs.shape[1]
        ksi = np.zeros((self.ns,self.ns))
        gamma = np.zeros((self.ns,T))

        for t in range(T-1):
            tmp_g = self.alpha[:,t] * self.beta[:,t]
            gamma[:,t] = self._normalize(tmp_g)

            tmp_k = self.A * np.dot(self.alpha[:,t],self.beta[:,t+1] * self.B[:,t+1].T)
            ksi += self._normalize(tmp_k)
         
        tmp_g = self.alpha[:,-1] * self.beta[:,-1]
        gamma[:,-1] = self._normalize(tmp_g)

        ksi = self._stochastize(ksi) 
        self.A = ksi # -ksi- is the expected state transition matrix

        #maximizing the states. i.e. mean vector and covariance
        expected_mu = np.zeros((self.nd,self.ns))
        expected_cov = np.zeros((self.nd,self.nd,self.ns))

        gamma_state_sum = np.sum(gamma,axis=1)
        gamma_state_sum += (gamma_state_sum==0)

        self.pi = self._normalize(gamma_state_sum/T) # expected prior distribution of state

        for s in range(self.ns):
            gamma_obs = obs * gamma[s]
            expected_mu[s] = np.sum(gamma_obs, axis=1) / gamma_state_sum[s]

            partial_covs = obs.T - expected_mu[s]
            partial_covs = np.dot(gamma,partial_covs*partial_covs) / gamma_state_sum
            expected_cov[s] = np.triu(partial_covs) + np.triu(partial_covs).T - np.diag(np.diag(partial_covs))
        
            
            
        #Ensure positive semidefinite by adding diagonal loading
        expected_cov += .01 * np.eye(self.nd)[:,:,None]
        print(np.linalg.det(expected_cov[0]),np.linalg.eig(expected_cov[0]))
        self.mu = expected_mu
        self.cov = expected_cov


    def train(self,obs,n_iter):
        flag = 2
        if(len(obs.shape) == 3):
            flag = 3
            self._initialize_state_gaussian(obs[0])
            for i in range(n_iter):
                print('training round: ',i)
                for o in obs:                
                    self._calculate_emission_probability(o)
                    self._forward()
                    self._backward()
                    self._Baum_Welch_Algorithm(o)

        elif(len(obs.shape) == 2):
            self._initialize_state_gaussian(obs)     
            for i in range(n_iter):
                print('training: ',i)
                self._calculate_emission_probability(obs)
                self._forward()
                self._backward()
                self._Baum_Welch_Algorithm(obs)
        else:
            raise Exception('Training Data Dimension Error; Expected 2d (num_states,*) or 3d (*,num_states,*)')

    def score(self,obs):        
        self._calculate_emission_probability(obs)
        log_likelihood = self._forward()
        return log_likelihood