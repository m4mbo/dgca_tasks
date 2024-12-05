import numpy as np  
import graph_tool.all as gt
from wrapt_timeout_decorator.wrapt_timeout_decorator import timeout
import matplotlib.pyplot as plt
from multiset import FrozenMultiset
from scipy.sparse.csgraph import connected_components


class Reservoir(object):

    def __init__(self, A: np.ndarray, S: np.ndarray, n_fixed: int):
        self.A = A      # N x N
        self.S = S      # N x S
        self.n_states = S.shape[1]
        self.n_fixed = n_fixed  # number of fixed I/O nodes

    def __str__(self) -> str:
        return f"Graph with {self.size()} nodes and {self.num_edges()} edges"

    def size(self) -> int:
        return self.A.shape[0]
    
    def num_edges(self) -> int:
        return np.sum(self.A)

    def connectivity(self) -> float:
        if self.size()>0:
            return self.num_edges() / self.size()**2
        else:
            return 0 # or could do np.nan?
    
    def get_neighbourhood(self) -> np.ndarray:
        """
        Returns matrix G
        Each row of G gives the neighbourhood information vector of one node in the graph
        """
        c_in = self.A.T @ self.S  # N x S
        c_out = self.A @ self.S   # N x S
        G = np.hstack([self.S, c_in, c_out])  # N x 3S
        bias_column = np.ones((G.shape[0], 1))  # N x 1
        return np.hstack([G, bias_column])  # N x (3S + 1)
    
    def to_edgelist(self) -> np.ndarray:
        """
        Returns an E*3 numpy array in the format
        source, target, weight (1s in this case)
        """
        es = np.nonzero(self.A)
        edge_list = np.array([es[0], es[1], self.A[es]]).T
        return edge_list
    
    def states_1d(self) -> list[int]:
        """
        Converts states out of one-hot encoding into a list of ints.
        eg. [[1,0,0,0],[0,0,1,0],[0,0,0,1]] -> [0,2,3]
        """
        return np.argmax(self.S, axis=1).tolist()
    
    def handle_IO_nodes(self, g: gt.Graph, 
                        pos: gt.VertexPropertyMap = None) -> gt.VertexPropertyMap:
        """
        Handles positioning of nodes, ensuring consistent spacing for input/output nodes
        and dynamic layout for other nodes. Also adjusts the outlines of I/O nodes.
        """
        # determine I/O nodes
        input_nodes = list(range(self.n_fixed // 2))
        output_nodes = list(range(self.n_fixed // 2, self.n_fixed))
        other_nodes = [v for v in g.vertices() if int(v) not in input_nodes + output_nodes]

        other_pos = gt.sfdp_layout(g, pos=pos)

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

        outline_color = g.new_vertex_property("vector<double>")
        pos = g.new_vertex_property("vector<double>")

        # assign positions and outline colors for input/output nodes
        for i, v in enumerate(input_nodes):
            pos[g.vertex(v)] = (input_x, center_y - total_height_input / 2 + i * spacing)
            outline_color[v] = [1, 0, 0, 1]  # red
        for i, v in enumerate(output_nodes):
            pos[g.vertex(v)] = (output_x, center_y - total_height_output / 2 + i * spacing)
            outline_color[v] = [0, 0, 1, 1]  # blue
        for v in other_nodes:
            pos[v] = other_pos[v]
            outline_color[v] = [0, 0, 0, 0]  # transparent

        g.vp['outline_color'] = outline_color
        g.vp['pos'] = pos
    

    def to_gt(self, basic: bool=False, 
              pos: gt.VertexPropertyMap=None) -> gt.Graph:
        """
        Converts it to a graph-tool graph.
        Good for visualisation and isomorphism checks.
        Nodes are coloured by state.
        Use basic=True if you just want the graph structure
        """
        n_nodes = self.size()
        edge_list = self.to_edgelist()
        g = gt.Graph(n_nodes)
        g.add_edge_list(edge_list, eprops=[("wgt", "double")])

        if not basic:
            # Assign node states as an internal property
            states = g.new_vertex_property('int', self.states_1d())
            g.vp['state'] = states

            # Assign colors based on states
            states_1d = self.states_1d()
            cmap = plt.get_cmap('viridis', self.n_states + 1)
            state_colors = cmap(states_1d)
            g.vp['plot_color'] = g.new_vertex_property('vector<double>', state_colors)
            # helper to set IO positions and colors
            self.handle_IO_nodes(g, pos=pos)

        return g

    
    def draw_gt(self, draw_edge_wgt: bool=False,
                pos: gt.VertexPropertyMap=None,
                interactive: bool=False,
                **kwargs) -> gt.VertexPropertyMap:
        """
        Draws a the graph using the graph-tool library 
        Relies on node and edge properties set by to_gt()
        Returns the node positions, which can then be passed in at the next
        call so that original nodes don't move to much if you are adding more etc.
        NB use output=filename to write to a file.
        """
        if self.size() == 0:
            print("Empty graph - can't draw")
            return None

        g = self.to_gt(pos=pos)

        # Use edge weights if enabled
        edge_pen_width = gt.prop_to_size(g.ep.wgt, mi=1, ma=7) if draw_edge_wgt else None

        # Draw the graph
        if interactive:
            pos_out = gt.interactive_window(
                g, pos=g.vp['pos'], vertex_fill_color=g.vp['plot_color'], 
                vertex_color=g.vp['outline_color'], 
                edge_pen_width=edge_pen_width, **kwargs
            )
        else:
            pos_out = gt.graph_draw(
                g, pos=g.vp['pos'], vertex_fill_color=g.vp['plot_color'], 
                vertex_color=g.vp['outline_color'], 
                edge_pen_width=edge_pen_width, **kwargs
            )

        return pos_out

    def state_hash(self) -> int:
        """
        Returns a hash of all the nieghbourhood state info.
        This is good as a preliminary isomorphism check (if
        two hashes are not the same then the graphs are definitely
        different).
        """
        return hash(FrozenMultiset(map(tuple, self.get_neighbourhood().tolist())))
    
    @timeout(5, use_signals=False)
    def is_isomorphic(self, other: "Reservoir") -> bool:
        """
        Checks if this graph is isomorhpic with another, conditional on node states.
        The decorator makes this function raise a timeout error if it takes longer than 5 seconds.
        """
        ne1 = self.num_edges()
        ne2 = other.num_edges()
        if ne1!=ne2:
            return False
        if ne1==0:
            # neither have any edges: structure doesn't matter so just check states match
            s1 = self.states_1d()
            s2 = other.states_1d()
            s1.sort()
            s2.sort()
            return s1==s2 
        gt1, gt2 = self.to_gt(), other.to_gt()
        # despite function name, subgraph=False does whole graph isomorphism
        # vertex_label param is used to condition the isomorphism on node state.
        mapping = gt.subgraph_isomorphism(gt1, gt2, 
                                          vertex_label=[gt1.vp.state, gt2.vp.state], 
                                          subgraph=False)
        # is isomorphic if at least one mapping was found
        return False if len(mapping)==0 else True

    def no_selfloops(self) -> "Reservoir":
        """
        Returns a copy of the graph in which all self-loops have been removed
        """
        out_A = self.A.copy()
        # set values on the diagonal to zero
        out_A[np.eye(out_A.shape[0], dtype=np.bool_)] = 0 
        return Reservoir(out_A, self.S.copy(), self.n_states)
    
    def get_components(self) -> tuple[np.ndarray]:
        """
        Returns a number for each node indicating which component
        it is part of.
        eg. [1,1,2,2,1] means nodes 0,1,4 form one connected component
        and nodes 2&3 form another.
        """
        # undirected for this purpose.
        _, cc = connected_components(self.A, directed=False)
        # count nodes in each component (will be sorted by component label 0->n_components)
        _, counts = np.unique(cc, return_counts=True)
        return cc, counts
    
    def get_largest_component_frac(self) -> float:
        """
        Returns the size of the largest component as a fraction of 
        the total number of nodes in the graph.
        """
        if self.size()==0:
            return 0
        else:
            _, component_sizes = self.get_components()
            return np.max(component_sizes) / self.size()

    def copy(self):
        return(np.copy(self.A), np.copy(self.S), self.n_fixed)
    
    def check_conditions(self, 
                         conditions: dict, 
                         verbose: bool=False) -> bool:
        size = self.size()
        conn = self.connectivity()
        frag = self.get_largest_component_frac()
        if 'max_size' in conditions:
            # should already be fine but double check
            if size > conditions['max_size']:
                if verbose:
                    print('Graph too big (should not happen!)')
                return False
        if 'min_size' in conditions:
            if size < conditions['min_size']:
                if verbose:
                    print('Graph too small.')
                return False
        if 'min_connectivity' in conditions:
            if conn < conditions['min_connectivity']:
                if verbose:
                    print('Graph too sparse')
                return False
        if 'max_connectivity' in conditions:
            if conn > conditions['max_connectivity']:
                if verbose:
                    print('Graph too dense')
                return False
        if 'min_component_frac' in conditions:
            if frag < conditions['min_component_frac']:
                if verbose:
                    print('Graph too fragmented')
                return False
        if verbose:
            print(f'Graph OK: size={size}, conn={conn*100:.2f}%, frag={frag:.2f}')
        return True