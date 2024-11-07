import numpy as np
from util import constants

class DGCA_M(object):
    def __init__(self, adj_mat: np.ndarray, state_mat, edge_mat):

        self.adj_mat = adj_mat  # N x N 
        self.state_mat = state_mat  # N x S
        self.edge_mat = edge_mat    # E
        self.n_states = state_mat.shape # S
        
        self.w_action = np.random.uniform(low=-1.0, high=1.0, size=(3*self.n_states + 1, 15))
        self.w_state = np.random.uniform(low=-1.0, high=1.0, size=(3*self.n_states + 1, self.n_states))
        
    def get_neighbourhood(self):
        """
        Returns matrix G
        Each row of G gives the neighbourhood information vector of one node in the graph
        """
        c_in = self.adj_mat.T @ self.state_mat  # N x S
        c_out = self.adj_mat @ self.state_mat   # N x S
        return np.hstack([self.state_mat, c_in, c_out], axis=1)  # N x 3S  
    
    def get_size(self):
        """
        Returns current number of nodes
        """
        return self.adj_mat.shape[0]
    
    def update_action(self, input):
        """
        First SLP
        """
        output = input @ self.w_action

        action = np.argmax(output[:4])
        to_delete = action == constants.ACTION["remove"]
        to_keep = action == 

    def update_state(self):
        """
        Second SLP
        """
        pass

    def update(self):
        
        pre = self.get_neighbourhood()
        self.update_action(pre)

        post = self.get_neighbourhood()
        self.update_state(post)

    def copy(self):
        """
        Returns a copy of this DGCA
        """
        out = DGCA_M(np.copy(self.adj_mat), np.copy(self.state_mat), np.copy(self.edge_mat))
        out.w_action = np.copy(self.w_action)
        out.w_state = np.copy(self.w_state)
        return out




