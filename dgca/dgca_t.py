import numpy as np
from util.consts import Q_B, Q_F, Q_M, Q_N
from dgca.reservoir import Reservoir

def onehot(x: np.ndarray):
    """
    Helper function to on hot encode an array x.
    """
    tf = x == np.max(x, axis=1, keepdims=True)
    return tf.astype(int)


class DGCA_T(object):
    def __init__(self, n_states: int=None):
        if not n_states:
            return
        self.w_action = np.random.uniform(low=-1.0, high=1.0, size=(3*n_states + 1, 15))
        self.w_state = np.random.uniform(low=-1.0, high=1.0, size=(3*n_states + 1, n_states))
    
    def update_action(self, res: Reservoir):
        """
        First SLP.
        """
        C = res.get_neighbourhood()
        D = C @ self.w_action   # N x 15

        # one hot in sections
        K = np.hstack((onehot(D[:,0:3]), onehot(D[:,3:7]), onehot(D[:,7:11]), onehot(D[:,11:15])))
        K = K.T 

        # action choices
        remove = K[0,:]
        noaction = K[1,:]
        divide = K[2,:]

        if res.n_io:
            remove[:res.n_io] = 0    # I/O nodes

        keep = np.hstack((np.logical_not(remove), divide)).astype(bool)
        
        # new node wiring
        none_f, k_fi, k_fa, k_ft = K[3, :], K[4, :], K[5, :], K[6, :]
        none_b, k_bi, k_ba, k_bt = K[7, :], K[8, :], K[9, :], K[10,:]
        none_n, k_ni, k_na, k_nt = K[11,:], K[12,:], K[13,:], K[14,:]

        none_all =  np.hstack((np.zeros((res.size())), np.logical_and(none_f, none_b, none_n))).astype(bool)
        keep = np.logical_and(keep, np.logical_not(none_all))

        I = np.eye(res.size())

        A, S = res.A, res.S
        A_new = np.kron(Q_M, A) \
            + np.kron(Q_F, (I @ np.diag(k_fi) + A @ np.diag(k_fa) + A.T @ np.diag(k_ft))) \
            + np.kron(Q_B, (np.diag(k_bi) @ I + np.diag(k_ba) @ A + np.diag(k_bt) @ A.T)) \
            + np.kron(Q_N, (np.diag(k_ni) @ I + np.diag(k_na) @ A + np.diag(k_nt) @ A.T))
        
        # keep only the nodes we need
        A_new = A_new[keep,:][:,keep]

        # duplicate relevant cols of state matrix
        S_new = np.vstack((S, S))
        S_new = S_new[keep,:]

        return Reservoir(A_new, S_new, res.n_io).no_islands()
       
    def update_state(self, res: Reservoir):
        """
        Second SLP.
        """
        G = res.get_neighbourhood()
        C = G @ self.w_state  # N x S
        return Reservoir(res.A, onehot(C), res.n_io)

    def step(self, res: Reservoir):
        """
        Pass through both SLPs. 
        """
        pre = self.update_action(res)
        post = self.update_state(pre)
        return post

    def copy(self):
        """
        Returns a copy of this DGCA.
        """
        out = DGCA_T(None)
        out.w_action = np.copy(self.w_action)
        out.w_state = np.copy(self.w_state)
        return out



