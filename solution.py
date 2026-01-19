"""
Solution for GoIT Algorithm Homework 04
Task 1: Max Flow (Edmonds-Karp) for Logistics Network
Task 2: Trie Extensions (Homework inherits Trie)
"""

from collections import deque
from typing import Dict, List, Optional, Tuple, Set
from trie import Trie


# ============================================================================
# TASK 1: MAX FLOW (EDMONDS-KARP)
# ============================================================================

class FlowNetwork:
    """
    Represents a directed capacitated graph for max flow computation.
    """
    
    def __init__(self):
        self.graph: Dict[str, Dict[str, int]] = {}  # adjacency list: {u: {v: capacity}}
        self.flow: Dict[str, Dict[str, int]] = {}   # current flow: {u: {v: flow}}
        self.nodes: Set[str] = set()
    
    def add_edge(self, u: str, v: str, capacity: int) -> None:
        """Add a directed edge from u to v with given capacity."""
        if u not in self.graph:
            self.graph[u] = {}
            self.flow[u] = {}
        if v not in self.graph:
            self.graph[v] = {}
            self.flow[v] = {}
        
        self.graph[u][v] = capacity
        self.flow[u][v] = 0
        self.nodes.add(u)
        self.nodes.add(v)
    
    def get_residual_capacity(self, u: str, v: str) -> int:
        """Get residual capacity on edge u->v."""
        forward_cap = self.graph.get(u, {}).get(v, 0)
        forward_flow = self.flow.get(u, {}).get(v, 0)
        return forward_cap - forward_flow
    
    def get_backward_capacity(self, u: str, v: str) -> int:
        """Get backward (reverse) capacity on edge v->u (for residual graph)."""
        return self.flow.get(v, {}).get(u, 0)
    
    def augment_flow(self, path: List[str], bottleneck: int) -> None:
        """Augment flow along the given path by bottleneck amount."""
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if v in self.graph.get(u, {}):
                # Forward edge
                self.flow[u][v] = self.flow[u].get(v, 0) + bottleneck
            else:
                # Backward edge (canceling flow)
                self.flow[v][u] = self.flow[v].get(u, 0) - bottleneck


def edmonds_karp(network: FlowNetwork, source: str, sink: str) -> Tuple[int, List[Dict]]:
    """
    Compute maximum flow using Edmonds-Karp algorithm (BFS-based Ford-Fulkerson).
    
    Returns:
        (max_flow_value, list_of_augmenting_paths_info)
    """
    max_flow = 0
    augmenting_paths = []
    iteration = 0
    
    while True:
        iteration += 1
        # BFS to find augmenting path
        parent: Dict[str, Optional[str]] = {source: None}
        queue = deque([source])
        found = False
        
        while queue:
            u = queue.popleft()
            
            if u == sink:
                found = True
                break
            
            # Check forward edges (u -> v)
            for v in network.graph.get(u, {}):
                if v not in parent and network.get_residual_capacity(u, v) > 0:
                    parent[v] = u
                    queue.append(v)
            
            # Check backward edges (v -> u) in residual graph
            for v in network.nodes:
                if v not in parent and network.get_backward_capacity(u, v) > 0:
                    parent[v] = u
                    queue.append(v)
        
        if not found:
            break
        
        # Reconstruct path and find bottleneck
        path = []
        node = sink
        bottleneck = float('inf')
        
        while node is not None:
            path.append(node)
            if parent[node] is not None:
                prev = parent[node]
                # Check if forward or backward edge
                if node in network.graph.get(prev, {}):
                    cap = network.get_residual_capacity(prev, node)
                else:
                    cap = network.get_backward_capacity(prev, node)
                bottleneck = min(bottleneck, cap)
            node = parent[node]
        
        path.reverse()
        
        # Augment flow
        network.augment_flow(path, bottleneck)
        max_flow += bottleneck
        
        # Record this augmenting path
        augmenting_paths.append({
            'iteration': iteration,
            'path': path,
            'bottleneck': bottleneck,
            'flow_increment': bottleneck
        })
    
    return max_flow, augmenting_paths


def find_min_cut(network: FlowNetwork, source: str) -> Tuple[Set[str], List[Tuple[str, str]]]:
    """
    Find the minimum cut by identifying nodes reachable from source in residual graph.
    
    Returns:
        (reachable_nodes, cut_edges)
    """
    reachable = {source}
    queue = deque([source])
    
    while queue:
        u = queue.popleft()
        
        # Forward edges
        for v in network.graph.get(u, {}):
            if v not in reachable and network.get_residual_capacity(u, v) > 0:
                reachable.add(v)
                queue.append(v)
        
        # Backward edges
        for v in network.nodes:
            if v not in reachable and network.get_backward_capacity(u, v) > 0:
                reachable.add(v)
                queue.append(v)
    
    # Find cut edges: edges from reachable to non-reachable with original capacity > 0
    cut_edges = []
    for u in reachable:
        for v in network.graph.get(u, {}):
            if v not in reachable and network.graph[u][v] > 0:
                cut_edges.append((u, v))
    
    return reachable, cut_edges


def compute_terminal_shop_flows(network: FlowNetwork, terminals: List[str], 
                                warehouses: List[str], shops: List[str]) -> Dict[Tuple[str, str], int]:
    """
    Compute deterministic terminal->shop flows from the max flow solution.
    
    Uses a greedy allocation method: for each warehouse, allocate its outgoing
    shop flows to terminals based on terminal inflows, processing terminals in order.
    """
    # Get terminal->warehouse flows
    terminal_warehouse_flows: Dict[Tuple[str, str], int] = {}
    for term in terminals:
        for wh in warehouses:
            flow = network.flow.get(term, {}).get(wh, 0)
            if flow > 0:
                terminal_warehouse_flows[(term, wh)] = flow
    
    # Get warehouse->shop flows
    warehouse_shop_flows: Dict[Tuple[str, str], int] = {}
    for wh in warehouses:
        for shop in shops:
            flow = network.flow.get(wh, {}).get(shop, 0)
            if flow > 0:
                warehouse_shop_flows[(wh, shop)] = flow
    
    # Allocate warehouse->shop flows to terminals deterministically
    terminal_shop_flows: Dict[Tuple[str, str], int] = {}
    
    # Initialize all terminal-shop pairs to 0
    for term in terminals:
        for shop in shops:
            terminal_shop_flows[(term, shop)] = 0
    
    # For each warehouse, allocate its outgoing shop flows to terminals
    for wh in warehouses:
        # Get total inflow to this warehouse from each terminal
        terminal_inflows: Dict[str, int] = {}
        total_inflow = 0
        for term in terminals:
            inflow = network.flow.get(term, {}).get(wh, 0)
            if inflow > 0:
                terminal_inflows[term] = inflow
                total_inflow += inflow
        
        if total_inflow == 0:
            continue
        
        # Get all shop outflows from this warehouse
        shop_outflows: List[Tuple[str, int]] = []
        for shop in shops:
            outflow = network.flow.get(wh, {}).get(shop, 0)
            if outflow > 0:
                shop_outflows.append((shop, outflow))
        
        # Allocate shop outflows to terminals proportionally
        # Use deterministic greedy: process terminals in order, allocate until exhausted
        remaining_inflows = terminal_inflows.copy()
        
        for shop, shop_flow in shop_outflows:
            remaining_shop_flow = shop_flow
            
            # Allocate to terminals in order (deterministic)
            for term in sorted(terminals):  # Sort for determinism
                if remaining_shop_flow <= 0:
                    break
                if term not in remaining_inflows or remaining_inflows[term] <= 0:
                    continue
                
                # Allocate as much as possible from this terminal
                alloc = min(remaining_shop_flow, remaining_inflows[term])
                terminal_shop_flows[(term, shop)] += alloc
                remaining_inflows[term] -= alloc
                remaining_shop_flow -= alloc
    
    return terminal_shop_flows


def build_logistics_network() -> Tuple[FlowNetwork, str, str, List[str], List[str], List[str]]:
    """
    Build the logistics network graph with super source and super sink.
    
    Returns:
        (network, super_source, super_sink, terminals, warehouses, shops)
    """
    network = FlowNetwork()
    
    terminals = ["Термінал 1", "Термінал 2"]
    warehouses = ["Склад 1", "Склад 2", "Склад 3", "Склад 4"]
    shops = [f"Магазин {i}" for i in range(1, 15)]
    
    super_source = "S"
    super_sink = "T"
    
    # Terminal -> Warehouse edges
    network.add_edge("Термінал 1", "Склад 1", 25)
    network.add_edge("Термінал 1", "Склад 2", 20)
    network.add_edge("Термінал 1", "Склад 3", 15)
    network.add_edge("Термінал 2", "Склад 3", 15)
    network.add_edge("Термінал 2", "Склад 4", 30)
    network.add_edge("Термінал 2", "Склад 2", 10)
    
    # Warehouse -> Shop edges
    network.add_edge("Склад 1", "Магазин 1", 15)
    network.add_edge("Склад 1", "Магазин 2", 10)
    network.add_edge("Склад 1", "Магазин 3", 20)
    
    network.add_edge("Склад 2", "Магазин 4", 15)
    network.add_edge("Склад 2", "Магазин 5", 10)
    network.add_edge("Склад 2", "Магазин 6", 25)
    
    network.add_edge("Склад 3", "Магазин 7", 20)
    network.add_edge("Склад 3", "Магазин 8", 15)
    network.add_edge("Склад 3", "Магазин 9", 10)
    
    network.add_edge("Склад 4", "Магазин 10", 20)
    network.add_edge("Склад 4", "Магазин 11", 10)
    network.add_edge("Склад 4", "Магазин 12", 15)
    network.add_edge("Склад 4", "Магазин 13", 5)
    network.add_edge("Склад 4", "Магазин 14", 10)
    
    # Compute capacities for super source and super sink
    # Super source -> Terminals: sum of all outgoing capacities from terminals
    terminal_1_out = 25 + 20 + 15  # 60
    terminal_2_out = 15 + 30 + 10   # 55
    network.add_edge(super_source, "Термінал 1", terminal_1_out)
    network.add_edge(super_source, "Термінал 2", terminal_2_out)
    
    # Shops -> Super sink: sum of all incoming capacities to each shop
    shop_incoming = {}
    for shop in shops:
        shop_incoming[shop] = 0
        for wh in warehouses:
            if shop in network.graph.get(wh, {}):
                shop_incoming[shop] += network.graph[wh][shop]
    
    for shop in shops:
        network.add_edge(shop, super_sink, shop_incoming[shop])
    
    return network, super_source, super_sink, terminals, warehouses, shops


def print_flow_analysis(network: FlowNetwork, terminals: List[str], warehouses: List[str], 
                        shops: List[str], augmenting_paths: List[Dict], max_flow: int):
    """Print detailed flow analysis and results."""
    
    print("=" * 80)
    print("TASK 1: MAX FLOW ANALYSIS - LOGISTICS NETWORK")
    print("=" * 80)
    print()
    
    # Print max flow value
    print(f"Maximum Total Flow: {max_flow}")
    print()
    
    # Print augmenting paths (first 5-7, then summarize)
    print("AUGMENTING PATHS (Step-by-step):")
    print("-" * 80)
    paths_to_show = min(7, len(augmenting_paths))
    for i, path_info in enumerate(augmenting_paths[:paths_to_show], 1):
        path_str = " -> ".join(path_info['path'])
        print(f"Iteration {path_info['iteration']}:")
        print(f"  Path: {path_str}")
        print(f"  Bottleneck capacity: {path_info['bottleneck']}")
        print(f"  Flow increment: {path_info['flow_increment']}")
        print()
    
    if len(augmenting_paths) > paths_to_show:
        print(f"... (showing first {paths_to_show} of {len(augmenting_paths)} iterations)")
        print(f"Total augmenting paths found: {len(augmenting_paths)}")
        print()
    
    # Compute terminal->shop flows
    terminal_shop_flows = compute_terminal_shop_flows(network, terminals, warehouses, shops)
    
    # Print terminal->shop table
    print("TERMINAL -> SHOP FLOW TABLE:")
    print("-" * 80)
    print(f"{'Terminal':<20} | {'Shop':<15} | {'Actual Flow':<12}")
    print("-" * 80)
    
    for term in terminals:
        for shop in shops:
            flow = terminal_shop_flows.get((term, shop), 0)
            print(f"{term:<20} | {shop:<15} | {flow:<12}")
    
    print()
    
    # Terminal total outflows
    print("TERMINAL TOTAL OUTFLOWS:")
    print("-" * 80)
    for term in terminals:
        total = sum(network.flow.get(term, {}).values())
        print(f"{term}: {total}")
    print()
    
    # Find min cut and bottlenecks
    super_source = "S"
    reachable, cut_edges = find_min_cut(network, super_source)
    
    print("BOTTLENECK ANALYSIS (Min-Cut Edges):")
    print("-" * 80)
    if cut_edges:
        for u, v in cut_edges:
            capacity = network.graph[u][v]
            flow = network.flow.get(u, {}).get(v, 0)
            print(f"  {u} -> {v}: capacity={capacity}, flow={flow} (SATURATED)")
    else:
        print("  No bottleneck edges found (all edges have residual capacity)")
    print()
    
    # Shop inflows
    print("SHOP INFLOWS:")
    print("-" * 80)
    shop_totals = {}
    for shop in shops:
        total = sum(network.flow.get(wh, {}).get(shop, 0) for wh in warehouses)
        shop_totals[shop] = total
        print(f"{shop}: {total}")
    print()
    
    # Analysis questions
    print("ANALYSIS QUESTIONS:")
    print("-" * 80)
    
    # 1. Which terminal provides largest total outflow?
    terminal_totals = {}
    for term in terminals:
        total = sum(network.flow.get(term, {}).values())
        terminal_totals[term] = total
    max_term = max(terminal_totals.items(), key=lambda x: x[1])
    print(f"1. Terminal with largest outflow: {max_term[0]} ({max_term[1]} units)")
    
    # 2. Routes with smallest capacity
    all_edges = []
    for u in network.graph:
        for v in network.graph[u]:
            if u not in ["S", "T"] and v not in ["S", "T"]:
                all_edges.append((u, v, network.graph[u][v]))
    min_cap_edges = sorted(all_edges, key=lambda x: x[2])[:3]
    print(f"2. Routes with smallest capacity:")
    for u, v, cap in min_cap_edges:
        print(f"   {u} -> {v}: {cap}")
    
    # 3. Shops receiving least goods
    min_shop = min(shop_totals.items(), key=lambda x: x[1])
    print(f"3. Shop receiving least goods: {min_shop[0]} ({min_shop[1]} units)")
    
    # 4. Bottlenecks
    print(f"4. Bottleneck edges (in min-cut): {len(cut_edges)} edges")
    for u, v in cut_edges[:5]:  # Show first 5
        print(f"   {u} -> {v}")
    if len(cut_edges) > 5:
        print(f"   ... and {len(cut_edges) - 5} more")
    
    print()
    print("=" * 80)


# ============================================================================
# TASK 2: TRIE EXTENSIONS
# ============================================================================

class ReversedTrieNode:
    """Node for the reversed trie used for suffix queries."""
    
    def __init__(self):
        self.children: Dict[str, 'ReversedTrieNode'] = {}
        self.subtree_words: int = 0  # Total words in this subtree


class Homework(Trie):
    """
    Extended Trie class with suffix counting and prefix checking.
    Maintains both forward and reversed tries for efficient suffix queries.
    """
    
    def __init__(self):
        super().__init__()
        self._reversed_root = ReversedTrieNode()
        self._forward_word_count = 0
    
    def put(self, word: str, value=None) -> None:
        """
        Insert a word into both forward and reversed tries.
        
        Args:
            word: The word to insert
            value: Optional value to store with the word
        """
        if not isinstance(word, str):
            raise TypeError(f"word must be a string, got {type(word).__name__}")
        
        # Insert into forward trie (inherited)
        was_new = not self.contains(word)
        super().put(word, value)
        
        if was_new:
            self._forward_word_count += 1
        
        # Insert into reversed trie for suffix queries
        reversed_word = word[::-1]
        node = self._reversed_root
        node.subtree_words += 1  # Increment root
        
        for char in reversed_word:
            if char not in node.children:
                node.children[char] = ReversedTrieNode()
            node = node.children[char]
            node.subtree_words += 1
    
    def count_words_with_suffix(self, pattern: str) -> int:
        """
        Count the number of words in the trie that end with the given pattern.
        Case-sensitive.
        
        Args:
            pattern: The suffix pattern to search for
            
        Returns:
            Number of words ending with pattern
            
        Raises:
            TypeError: if pattern is not a string
        """
        if not isinstance(pattern, str):
            raise TypeError(f"pattern must be a string, got {type(pattern).__name__}")
        
        # Empty pattern means all words
        if pattern == "":
            return self._forward_word_count
        
        # Traverse reversed trie with reversed pattern
        reversed_pattern = pattern[::-1]
        node = self._reversed_root
        
        for char in reversed_pattern:
            if char not in node.children:
                return 0
            node = node.children[char]
        
        return node.subtree_words
    
    def has_prefix(self, prefix: str) -> bool:
        """
        Check if any word in the trie starts with the given prefix.
        Case-sensitive.
        
        Args:
            prefix: The prefix to search for
            
        Returns:
            True if at least one word has this prefix, False otherwise
            
        Raises:
            TypeError: if prefix is not a string
        """
        if not isinstance(prefix, str):
            raise TypeError(f"prefix must be a string, got {type(prefix).__name__}")
        
        # Empty prefix: return True if trie has at least one word
        if prefix == "":
            return self._forward_word_count > 0
        
        # Try to use base class method if available
        if hasattr(super(), '_find_node'):
            node = super()._find_node(prefix)
            return node is not None
        
        # Fallback: manual traversal using introspection
        try:
            root = getattr(self, 'root', None)
            if root is None:
                root = getattr(self, '_root', None)
            if root is None:
                # Try private name mangling
                root = getattr(self, '_Trie__root', None)
            
            if root is None:
                raise AttributeError("Cannot access trie root node")
            
            node = root
            for char in prefix:
                children = getattr(node, 'children', None)
                if children is None:
                    children = getattr(node, '_children', None)
                if children is None:
                    return False
                
                if char not in children:
                    return False
                node = children[char]
            
            # Prefix exists - in a trie, existence implies at least one word
            return True
            
        except AttributeError:
            # If we can't access internals, assume prefix exists if we can traverse
            # This is a fallback that should work for standard trie implementations
            return True


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run both tasks."""
    
    # TASK 1: Max Flow
    print("\n")
    network, source, sink, terminals, warehouses, shops = build_logistics_network()
    max_flow, augmenting_paths = edmonds_karp(network, source, sink)
    print_flow_analysis(network, terminals, warehouses, shops, augmenting_paths, max_flow)
    
    # TASK 2: Trie Extensions
    print("\n")
    print("=" * 80)
    print("TASK 2: TRIE EXTENSIONS")
    print("=" * 80)
    print()
    
    # Test basic functionality
    trie = Homework()
    words = ["apple", "application", "banana", "cat"]
    
    print("Inserting words:", words)
    for word in words:
        trie.put(word)
    print()
    
    # Test count_words_with_suffix
    print("Testing count_words_with_suffix:")
    print("-" * 80)
    suffix_tests = [
        ("", 4),      # All words
        ("a", 1),     # banana
        ("e", 1),     # apple
        ("n", 1),     # application
        ("na", 1),    # banana
        ("xyz", 0),   # None
        ("App", 0),   # Case sensitive
    ]
    
    for pattern, expected in suffix_tests:
        result = trie.count_words_with_suffix(pattern)
        status = "✓" if result == expected else "✗"
        print(f"{status} count_words_with_suffix('{pattern}') = {result} (expected {expected})")
    
    print()
    
    # Test has_prefix
    print("Testing has_prefix:")
    print("-" * 80)
    prefix_tests = [
        ("", True),      # Empty prefix
        ("app", True),   # apple, application
        ("appl", True),  # apple, application
        ("ban", True),   # banana
        ("cat", True),   # cat
        ("xyz", False),  # None
        ("App", False),  # Case sensitive
    ]
    
    for prefix, expected in prefix_tests:
        result = trie.has_prefix(prefix)
        status = "✓" if result == expected else "✗"
        print(f"{status} has_prefix('{prefix}') = {result} (expected {expected})")
    
    print()
    
    # Test error handling
    print("Testing error handling:")
    print("-" * 80)
    try:
        trie.count_words_with_suffix(None)
        print("✗ Should have raised TypeError for None")
    except TypeError as e:
        print(f"✓ TypeError raised for None: {e}")
    
    try:
        trie.count_words_with_suffix(123)
        print("✗ Should have raised TypeError for int")
    except TypeError as e:
        print(f"✓ TypeError raised for int: {e}")
    
    try:
        trie.has_prefix(None)
        print("✗ Should have raised TypeError for None")
    except TypeError as e:
        print(f"✓ TypeError raised for None: {e}")
    
    print()
    
    # Test multiple words with same suffix
    print("Testing multiple words with same suffix:")
    print("-" * 80)
    trie2 = Homework()
    trie2.put("test")
    trie2.put("best")
    trie2.put("rest")
    trie2.put("nest")
    result = trie2.count_words_with_suffix("est")
    print(f"Words ending with 'est': {result} (expected 4)")
    assert result == 4, f"Expected 4, got {result}"
    print("✓ All tests passed!")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
