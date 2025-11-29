"""
Graph Edit Distance Calculator Module
Implements GED with mBERT-based semantic costs for multilingual UML assessment
"""

import networkx as nx
import logging
from typing import Dict, Any, List, Tuple, Optional
from modules.mbert_processor import MBERTProcessor, calculate_attribute_similarity, calculate_method_similarity


class GEDCalculator:
    """Calculate Graph Edit Distance with semantic-aware cost functions"""
    
    def __init__(self, mbert_processor: MBERTProcessor, config: Dict[str, Any]):
        """
        Initialize GED calculator
        
        Args:
            mbert_processor: MBERTProcessor for semantic similarity
            config: Configuration dictionary with weights and penalties
        """
        self.mbert_processor = mbert_processor
        self.config = config
        
        # Extract configuration
        assessment_config = config.get('assessment', {})
        self.weights = assessment_config.get('weights', {
            'class_name': 0.5,
            'attributes': 0.25,
            'methods': 0.25
        })
        
        self.penalties = assessment_config.get('penalties', {
            'node_deletion': 1.0,
            'node_insertion': 1.0,
            'edge_deletion': 0.5,
            'edge_insertion': 0.5,
            'wrong_relationship': 0.8
        })
        
        self.semantic_threshold = assessment_config.get('semantic_threshold', 0.85)
        
        # Track edit operations for detailed analysis
        self.edit_operations = []
    
    def calculate_ged(self, G_key: nx.DiGraph, G_student: nx.DiGraph, 
                     timeout: int = 60) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate Graph Edit Distance with mBERT-based semantic costs
        
        GED(G1, G2) = min Σ c(e_i) over all edit sequences
        
        Args:
            G_key: Key reference graph
            G_student: Student submission graph
            timeout: Timeout in seconds for GED calculation
        
        Returns:
            Tuple of (ged_value, detailed_breakdown)
        """
        self.edit_operations = []
        
        logging.info(f"Calculating GED between {G_key.graph['diagram_id']} and {G_student.graph['diagram_id']}")
        
        # Define node substitution cost function
        def node_subst_cost(node1, node2):
            return self._node_substitution_cost(G_key, G_student, node1, node2)
        
        # Define node deletion cost function
        def node_del_cost(node):
            return self.penalties['node_deletion']
        
        # Define node insertion cost function
        def node_ins_cost(node):
            return self.penalties['node_insertion']
        
        # Define edge substitution cost function
        def edge_subst_cost(edge1, edge2):
            return self._edge_substitution_cost(G_key, G_student, edge1, edge2)
        
        # Define edge deletion cost function
        def edge_del_cost(edge):
            return self.penalties['edge_deletion']
        
        # Define edge insertion cost function
        def edge_ins_cost(edge):
            return self.penalties['edge_insertion']
        
        try:
            # Calculate GED using NetworkX
            # Note: This is an approximation algorithm for efficiency
            ged_value = nx.graph_edit_distance(
                G_key,
                G_student,
                node_subst_cost=node_subst_cost,
                node_del_cost=node_del_cost,
                node_ins_cost=node_ins_cost,
                edge_subst_cost=edge_subst_cost,
                edge_del_cost=edge_del_cost,
                edge_ins_cost=edge_ins_cost,
                timeout=timeout
            )
            
            logging.info(f"GED calculated: {ged_value:.4f}")
            
        except nx.NetworkXError as e:
            logging.error(f"Error calculating GED: {e}")
            # Fallback: use simple node and edge count differences
            ged_value = self._fallback_ged_calculation(G_key, G_student)
            logging.warning(f"Using fallback GED calculation: {ged_value:.4f}")
        
        # Generate detailed breakdown
        breakdown = self._generate_breakdown(G_key, G_student, ged_value)
        
        return ged_value, breakdown
    
    def _node_substitution_cost(self, G_key: nx.DiGraph, G_student: nx.DiGraph, 
                                node1: str, node2: str) -> float:
        """
        Calculate cost of substituting node1 with node2
        
        Args:
            G_key: Key graph
            G_student: Student graph
            node1: Node ID in key graph
            node2: Node ID in student graph
        
        Returns:
            Substitution cost
        """
        # Get node attributes
        node1_attrs = G_key.nodes.get(node1)
        node2_attrs = G_student.nodes.get(node2)
        
        if node1_attrs is None or node2_attrs is None:
            return self.penalties['node_deletion']  # Treat as deletion/insertion
        
        # Calculate semantic cost for class name
        name1 = node1_attrs['name']
        name2 = node2_attrs['name']
        name_cost = self.mbert_processor.calculate_semantic_cost(name1, name2)
        
        # Calculate attribute similarity
        attr_cost = calculate_attribute_similarity(
            node1_attrs.get('attributes', []),
            node2_attrs.get('attributes', []),
            self.mbert_processor
        )
        
        # Calculate method similarity
        method_cost = calculate_method_similarity(
            node1_attrs.get('methods', []),
            node2_attrs.get('methods', []),
            self.mbert_processor
        )
        
        # Weighted combination
        total_cost = (
            self.weights['class_name'] * name_cost +
            self.weights['attributes'] * attr_cost +
            self.weights['methods'] * method_cost
        )
        
        # Log operation if cost is significant
        if total_cost > 0.1:
            self.edit_operations.append({
                'type': 'node_substitution',
                'node1': node1,
                'node2': node2,
                'name1': name1,
                'name2': name2,
                'cost': total_cost,
                'name_cost': name_cost,
                'attr_cost': attr_cost,
                'method_cost': method_cost
            })
        
        return total_cost
    
    def _edge_substitution_cost(self, G_key: nx.DiGraph, G_student: nx.DiGraph,
                                edge1: Tuple[str, str], edge2: Tuple[str, str]) -> float:
        """
        Calculate cost of substituting edge1 with edge2
        
        Args:
            G_key: Key graph
            G_student: Student graph
            edge1: Edge tuple (source, target) in key graph
            edge2: Edge tuple (source, target) in student graph
        
        Returns:
            Substitution cost
        """
        # Get edge attributes
        edge1_attrs = G_key.edges.get(edge1)
        edge2_attrs = G_student.edges.get(edge2)
        
        if edge1_attrs is None or edge2_attrs is None:
            return self.penalties['edge_deletion']  # Treat as deletion/insertion
        
        # Check if relationship types match
        rel_type1 = edge1_attrs.get('rel_type', '')
        rel_type2 = edge2_attrs.get('rel_type', '')
        
        if rel_type1 == rel_type2:
            cost = 0.0  # Correct relationship type
        else:
            cost = self.penalties['wrong_relationship']
            
            # Log wrong relationship
            self.edit_operations.append({
                'type': 'edge_wrong_type',
                'edge1': edge1,
                'edge2': edge2,
                'rel_type1': rel_type1,
                'rel_type2': rel_type2,
                'cost': cost
            })
        
        return cost
    
    def _fallback_ged_calculation(self, G_key: nx.DiGraph, G_student: nx.DiGraph) -> float:
        """
        Fallback GED calculation using simple heuristics
        
        Args:
            G_key: Key graph
            G_student: Student graph
        
        Returns:
            Approximate GED value
        """
        # Count node differences
        num_nodes_key = G_key.number_of_nodes()
        num_nodes_student = G_student.number_of_nodes()
        node_diff = abs(num_nodes_key - num_nodes_student)
        
        # Count edge differences
        num_edges_key = G_key.number_of_edges()
        num_edges_student = G_student.number_of_edges()
        edge_diff = abs(num_edges_key - num_edges_student)
        
        # Estimate cost
        ged_estimate = (
            node_diff * self.penalties['node_deletion'] +
            edge_diff * self.penalties['edge_deletion']
        )
        
        return ged_estimate
    
    def _generate_breakdown(self, G_key: nx.DiGraph, G_student: nx.DiGraph, 
                           ged_value: float) -> Dict[str, Any]:
        """
        Generate detailed breakdown of the assessment
        
        Args:
            G_key: Key graph
            G_student: Student graph
            ged_value: Calculated GED value
        
        Returns:
            Dictionary with detailed metrics
        """
        breakdown = {
            'ged_value': ged_value,
            'num_nodes_key': G_key.number_of_nodes(),
            'num_nodes_student': G_student.number_of_nodes(),
            'num_edges_key': G_key.number_of_edges(),
            'num_edges_student': G_student.number_of_edges(),
            'node_difference': abs(G_key.number_of_nodes() - G_student.number_of_nodes()),
            'edge_difference': abs(G_key.number_of_edges() - G_student.number_of_edges()),
            'edit_operations': self.edit_operations.copy()
        }
        
        # Analyze edit operations
        breakdown['num_node_substitutions'] = sum(
            1 for op in self.edit_operations if op['type'] == 'node_substitution'
        )
        breakdown['num_edge_wrong_type'] = sum(
            1 for op in self.edit_operations if op['type'] == 'edge_wrong_type'
        )
        
        # Calculate component costs
        total_name_cost = sum(
            op.get('name_cost', 0) for op in self.edit_operations 
            if op['type'] == 'node_substitution'
        )
        total_attr_cost = sum(
            op.get('attr_cost', 0) for op in self.edit_operations 
            if op['type'] == 'node_substitution'
        )
        total_method_cost = sum(
            op.get('method_cost', 0) for op in self.edit_operations 
            if op['type'] == 'node_substitution'
        )
        
        breakdown['total_name_cost'] = total_name_cost
        breakdown['total_attr_cost'] = total_attr_cost
        breakdown['total_method_cost'] = total_method_cost
        
        return breakdown
    
    def calculate_optimized_ged(self, G_key: nx.DiGraph, G_student: nx.DiGraph) -> Tuple[float, Dict[str, Any]]:
        """
        Optimized GED calculation using greedy matching
        Faster but less accurate than full GED
        
        Args:
            G_key: Key graph
            G_student: Student graph
        
        Returns:
            Tuple of (ged_value, breakdown)
        """
        self.edit_operations = []
        total_cost = 0.0
        
        # Match nodes using greedy approach
        key_nodes = list(G_key.nodes())
        student_nodes = list(G_student.nodes())
        
        matched_pairs = []
        used_student_nodes = set()
        
        # For each key node, find best matching student node
        for key_node in key_nodes:
            best_match = None
            best_cost = float('inf')
            
            for student_node in student_nodes:
                if student_node in used_student_nodes:
                    continue
                
                cost = self._node_substitution_cost(G_key, G_student, key_node, student_node)
                
                if cost < best_cost:
                    best_cost = cost
                    best_match = student_node
            
            if best_match is not None and best_cost < self.penalties['node_deletion']:
                matched_pairs.append((key_node, best_match))
                used_student_nodes.add(best_match)
                total_cost += best_cost
            else:
                # Node deletion
                total_cost += self.penalties['node_deletion']
        
        # Add insertion cost for unmatched student nodes
        num_insertions = len(student_nodes) - len(used_student_nodes)
        total_cost += num_insertions * self.penalties['node_insertion']
        
        # Match edges
        for key_edge in G_key.edges():
            # Find corresponding edge in student graph
            key_src, key_tgt = key_edge
            
            # Find matched student nodes
            student_src = None
            student_tgt = None
            
            for k_node, s_node in matched_pairs:
                if k_node == key_src:
                    student_src = s_node
                if k_node == key_tgt:
                    student_tgt = s_node
            
            if student_src and student_tgt:
                student_edge = (student_src, student_tgt)
                
                if G_student.has_edge(student_src, student_tgt):
                    edge_cost = self._edge_substitution_cost(G_key, G_student, key_edge, student_edge)
                    total_cost += edge_cost
                else:
                    # Edge deletion
                    total_cost += self.penalties['edge_deletion']
            else:
                # Edge deletion (nodes not matched)
                total_cost += self.penalties['edge_deletion']
        
        # Add insertion cost for extra student edges
        student_edges_count = 0
        for student_edge in G_student.edges():
            student_src, student_tgt = student_edge
            
            # Check if both nodes were matched
            if student_src in used_student_nodes and student_tgt in used_student_nodes:
                # Check if corresponding key edge exists
                key_src = None
                key_tgt = None
                
                for k_node, s_node in matched_pairs:
                    if s_node == student_src:
                        key_src = k_node
                    if s_node == student_tgt:
                        key_tgt = k_node
                
                if key_src and key_tgt:
                    if not G_key.has_edge(key_src, key_tgt):
                        total_cost += self.penalties['edge_insertion']
                        student_edges_count += 1
        
        breakdown = self._generate_breakdown(G_key, G_student, total_cost)
        breakdown['method'] = 'greedy_matching'
        breakdown['matched_pairs'] = len(matched_pairs)
        
        return total_cost, breakdown
