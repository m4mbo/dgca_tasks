import numpy as np
from util.consts import Q_B, Q_F, Q_M, Q_N
from dgca.reservoir import Reservoir

def one_hot(x: np.ndarray):
    """
    Helper function to on hot encode an array x.
    """
    tf = x == np.max(x, axis=0, keepdims=True)
    return tf.astype(int)


class DGCA_M(object):
    def __init__(self, n_states: int=None):
        if not n_states:
            return
        self.w_action = np.random.uniform(low=-1.0, high=1.0, size=(3*n_states + 1, 15))
        self.w_state = np.random.uniform(low=-1.0, high=1.0, size=(3*n_states + 1, n_states))
    
    def update_action(self, res: Reservoir):
        """
        First SLP.
        """
        G = res.get_neighbourhood()
        C = (G @ self.w_action).T   # 15 x N
        K = one_hot(C) 
        
        remove = K[0]
        _ = K[1]                    # no action
        divide = K[2]

        remove[:res.n_fixed] = 0    # I/O nodes
        divide[:res.n_fixed] = 0    # I/O nodes

        keep = np.hstack((np.logical_not(remove), divide)).astype(bool)

        # new node wiring
        fe = K[3:7]                 # from existing
        te = K[7:11]                # to existing
        tn = K[11:]                 # to new

        _, k_fi, k_fa, k_ft = fe[0], fe[1], fe[2], fe[3]
        _, k_bi, k_ba, k_bt = te[0], te[1], te[2], te[3]
        _, k_ni, k_na, k_nt = tn[0], tn[1], tn[2], tn[3]

        A = res.A
        I = np.eye(res.size())

        upper_left = np.kron(Q_M, A)   # remains untouched
        upper_right = np.kron(Q_F, (I @ np.diag(k_fi) + A @ np.diag(k_fa) + A.T @ np.diag(k_ft)))
        lower_left = np.kron(Q_B, (np.diag(k_bi) @ I + np.diag(k_ba) @ A + np.diag(k_bt) @ A.T))
        lower_right = np.kron(Q_N, (np.diag(k_ni) @ I + np.diag(k_na) @ A + np.diag(k_nt) @ A.T))

        A_new = upper_left + upper_right + lower_left + lower_right     # 2N x 2N, assuming all nodes divide
        A_new = A_new[keep,:][:,keep]

        S_new = np.vstack((res.S, res.S))[keep,:] 
        
        return Reservoir(A_new, S_new, res.n_fixed) 
       
    def update_state(self, res: Reservoir):
        """
        Second SLP.
        """
        G = res.get_neighbourhood()
        C = G @ self.w_state  # N x S
        return Reservoir(res.A, one_hot(C), res.n_fixed)

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
        out = DGCA_M(None)
        out.w_action = np.copy(self.w_action)
        out.w_state = np.copy(self.w_state)
        return out



