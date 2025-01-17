import numpy as np  
import graph_tool.all as gt
from util.consts import POLAR_TABLE
from grow.graph import GraphDef
from scipy.sparse.csgraph import connected_components
import sklearn.linear_model as linfit
import matplotlib.pyplot as plt

def check_conditions(res: "Reservoir",
                     conditions: dict, 
                     verbose: bool=False) -> bool:
    size = res.size()
    conn = res.connectivity()
    frag = res.get_largest_component_frac()
    if 'max_size' in conditions:
        # should already be fine but double check
        if size > conditions['max_size']:
            if verbose:
                print('Reservoir too big (should not happen!)')
            return False
    if 'io_path' in conditions:
        assert res.input_nodes and res.output_nodes
        if not res.io_path() and conditions['io_path']:
            if verbose:
                print('No I/O path.')
            return False
    if 'min_size' in conditions:
        if size < conditions['min_size']:
            if verbose:
                print('Reservoir too small.')
            return False
    if 'min_connectivity' in conditions:
        if conn < conditions['min_connectivity']:
            if verbose:
                print('Reservoir too sparse')
            return False
    if 'max_connectivity' in conditions:
        if conn > conditions['max_connectivity']:
            if verbose:
                print('Reservoir too dense')
            return False
    if 'min_component_frac' in conditions:
        if frag < conditions['min_component_frac']:
            if verbose:
                print('Reservoir too fragmented')
            return False
    if verbose:
        print(f'Reservoir OK: size={size}, conn={conn*100:.2f}%, frag={frag:.2f}')
    return True

def dfs_directed(A: np.ndarray, current: int, visited: set) -> bool:
    """
    Perform a recursive DFS on a directed adjacency matrix
    """
    if A.shape[0] == 0:
        return False
    # current node as visited
    visited.add(current) 
    # visit neighbors
    neighbors = np.nonzero(A[current])[0]  # directed neighbors
    for neighbor in neighbors:
        if neighbor not in visited:
            if dfs_directed(A, neighbor, visited):
                return True
    return False


class Reservoir(GraphDef):

    def __init__(self, A: np.ndarray, S: np.ndarray, input_nodes: int=0, output_nodes: int=0):
        super().__init__(A, S)
        self.input_nodes = input_nodes  # number of fixed I/O nodes
        self.output_nodes = output_nodes 

    def pp(self, 
           g: gt.Graph, 
           pos: gt.VertexPropertyMap = None) -> gt.VertexPropertyMap:
        """
        Pretty prints the input/output nodes.
        Handles positioning of nodes, ensuring consistent spacing for I/O nodes
        and dynamic layout for other nodes. Also adjusts the outlines of I/O nodes.
        """
        # assign colors based on states
        states_1d = self.states_1d()
        cmap = plt.get_cmap('viridis', self.n_states + 1)
        state_colors = cmap(states_1d)
        g.vp['plot_color'] = g.new_vertex_property('vector<double>', state_colors)
        
        # determine I/O nodes
        input_nodes = list(range(self.input_nodes)) if self.input_nodes > 0 else []
        output_nodes = list(range(self.input_nodes, self.input_nodes+self.output_nodes)) if self.output_nodes > 0 else []
        other_nodes = [v for v in g.vertices() if int(v) not in input_nodes + output_nodes]

        # use sfdp_layout for the general layout
        other_pos = gt.sfdp_layout(g, pos=pos)

        # initialize vertex properties
        outline_color = g.new_vertex_property("vector<double>")
        pos = g.new_vertex_property("vector<double>")

        if not input_nodes and not output_nodes:
            # no I/O nodes
            for v in g.vertices():
                pos[v] = other_pos[v]
                outline_color[v] = [0, 0, 0, 0]  # default outline color (transparent)
        else:
            # bounding box of other nodes
            x_min, x_max = float("inf"), float("-inf")
            y_min, y_max = float("inf"), float("-inf")
            for v in other_nodes:
                x, y = other_pos[v]
                x_min = min(x_min, x)
                x_max = max(x_max, x)
                y_min = min(y_min, y)
                y_max = max(y_max, y)

            # handle case where y_min == y_max, probably only one node
            if y_min == y_max:
                y_min -= 1
                y_max += 1

            # dynamic offsets
            input_x = x_min - 2.0  # left
            output_x = x_max + 2.0  # right
            spacing = max(abs(y_max - y_min) / max(len(input_nodes), len(output_nodes)), 1.0)  # Vertical spacing

            # center nodes vertically around graph middle
            center_y = (y_min + y_max) / 2.0
            total_height_input = spacing * (len(input_nodes) - 1)
            total_height_output = spacing * (len(output_nodes) - 1)

            # assign positions and outline colors for input/output nodes
            for i, v in enumerate(input_nodes):
                pos[g.vertex(v)] = (input_x, center_y - total_height_input / 2 + i * spacing)
                outline_color[v] = [1, 1, 1, 0.8]  # red
            for i, v in enumerate(output_nodes):
                pos[g.vertex(v)] = (output_x, center_y - total_height_output / 2 + i * spacing)
                outline_color[v] = [1, 1, 1, 0.8]  # blue
            for v in other_nodes:
                pos[v] = other_pos[v]
                outline_color[v] = [0, 0, 0, 0]  # transparent

        # Assign edge colors based on weights
        edge_colors = g.new_edge_property("vector<double>")
        for e in g.edges():
            weight = g.ep.wgt[e]  # Assuming weights are stored as edge property 'wgt'
            if weight > 0:
                edge_colors[e] = [0, 0, 0, 1]  # Black for positive weights
            else:
                edge_colors[e] = [1, 0, 0, 1]  # Red for negative weights

        # Assign properties to the graph
        g.vp['outline_color'] = outline_color
        g.vp['pos'] = pos
        g.ep['edge_color'] = edge_colors
    
    def io_path(self) -> bool:
        """
        Check if there is a path from every input node to at least one output node.
        """
        input_nodes = range(self.input_nodes)
        output_nodes = range(self.input_nodes, self.input_nodes+self.output_nodes)
        
        for input_node in input_nodes:
            visited = set()
            # dfs from input
            dfs_directed(self.A, input_node, visited)
            if not any(output_node in visited for output_node in output_nodes):
                return False
        return True
    
    def bipolar(self) -> "Reservoir":
        """
        Return a copy of the graph with weights converted to bipolar (-1,1) from one-hot encoding.
        """
        node_states = np.array(self.states_1d())
        
        # row and column indices of edges
        rows, cols = np.nonzero(self.A)

        # map the states of the source and target nodes 
        states_from = node_states[rows]
        states_to = node_states[cols]
        
        # vectorize to weights
        new_weights = POLAR_TABLE[states_from, states_to]
        
        A_new = np.zeros_like(self.A)
        A_new[rows, cols] = new_weights  

        return Reservoir(A_new, self.S, self.input_nodes, self.output_nodes)
    
    def no_islands(self) -> "Reservoir":
        """
        Returns a copy of the graph in which all isolated group of nodes 
        (relative to the input nodes) have been removed
        """
        # no input nodes
        if (self.input_nodes+self.output_nodes == 0) or self.size() == 0:
            return self.copy()
        
        # only one big chunk of cc
        n_cc, _ = connected_components(self.A, directed=False)
        if n_cc == 1:
            return self.copy()
                
        input_nodes = range(self.input_nodes)
        reachable_mask = np.zeros(self.A.shape[0], dtype=bool)
        
        # dfs from each input node
        for input_node in input_nodes:
            visited = set()
            dfs_directed(self.A, input_node, visited)
            reachable_mask[list(visited)] = True

        # I/O nodes are protected
        for node in range(self.input_nodes+self.output_nodes):
            reachable_mask[node] = True
    
        final_A = self.A[reachable_mask][:, reachable_mask]
        final_S = self.S[reachable_mask]
        return Reservoir(final_A, final_S, self.input_nodes, self.output_nodes)

    def fit(self, u, inputgain, feedbackgain, y_train=None):
        """
        Fits a reservoir computing model using Bayesian Ridge Regression.

        Parameters:
        - u (input_dim, time_steps): Input data.
        - inputgain: Scaling factor for w_in.
        - feedbackgain: Scaling factor for w_res.
        - y_train (1, time_steps): Target data.
        
        Returns:
        - w_out (reservoir_dim + 1, 1): Output weight matrix.
        - reservoir_state (reservoir_dim + 1, time_steps): State matrix.
        """
        w_in = np.random.randint(-1, 2, (1, self.size()))
        if self.input_nodes:
            w_in[:, self.input_nodes:] = 0

        reservoir_state = np.zeros((self.size(), u.shape[1]))
        w_res = self.A

        for i in range(u.shape[1]):
            if i == 0:
                reservoir_state[:,i] = np.tanh(inputgain*w_in.T @ u[:,i])
            else:
                reservoir_state[:,i] = np.tanh(inputgain*w_in.T @ u[:,i] + feedbackgain*w_res.T @ reservoir_state[:,i-1])
        
        if self.output_nodes:
            reservoir_state = reservoir_state[self.input_nodes:self.input_nodes+self.output_nodes] # keeping only output nodes
        
        # add bias node
        reservoir_state = np.concatenate((reservoir_state, np.ones((1, reservoir_state.shape[1]))),axis=0)

        w_out = None
        if y_train is not None:
            # Tikhonov regularization to fit and generalize on unseen data
            regression = linfit.BayesianRidge(max_iter=3000, tol=1e-6, verbose=False, fit_intercept=False)
            # remove first element as it does not consider previous state
            regression.fit(reservoir_state[:,1:].T, y_train[0,1:])
            w_out = regression.coef_
        return w_out, reservoir_state
    
    def copy(self):
        return Reservoir(np.copy(self.A), np.copy(self.S), self.input_nodes, self.output_nodes)


    