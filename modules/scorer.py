"""
Scorer Module
Converts GED to similarity scores and generates component scores
"""

import numpy as np
import logging
from typing import Dict, Any, List


class Scorer:
    """Calculate and normalize similarity scores"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize scorer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
    
    def calculate_similarity_score(self, ged_value: float, 
                                   max_possible_ged: float = None) -> float:
        """
        Convert GED to similarity score (0-100)
        
        Score = 1 / (1 + GED) × 100
        
        Or with normalization:
        Score = 1 / (1 + GED/max_GED) × 100
        
        Args:
            ged_value: Graph Edit Distance value
            max_possible_ged: Maximum possible GED for normalization
        
        Returns:
            Similarity score between 0 and 100
        """
        if max_possible_ged is not None and max_possible_ged > 0:
            normalized_ged = ged_value / max_possible_ged
        else:
            normalized_ged = ged_value
        
        # Calculate score using inverse formula
        score = (1 / (1 + normalized_ged)) * 100
        
        return score
    
    def estimate_max_ged(self, num_nodes_key: int, num_edges_key: int, 
                        penalties: Dict[str, float]) -> float:
        """
        Estimate maximum possible GED
        (all nodes and edges deleted/inserted)
        
        Args:
            num_nodes_key: Number of nodes in key diagram
            num_edges_key: Number of edges in key diagram
            penalties: Penalty configuration
        
        Returns:
            Estimated maximum GED
        """
        # Assume worst case: delete all key elements and insert completely different ones
        max_ged = (
            num_nodes_key * penalties.get('node_deletion', 1.0) +
            num_nodes_key * penalties.get('node_insertion', 1.0) +
            num_edges_key * penalties.get('edge_deletion', 0.5) +
            num_edges_key * penalties.get('edge_insertion', 0.5)
        )
        
        return max_ged
    
    def calculate_component_scores(self, breakdown: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate separate scores for different components
        
        Args:
            breakdown: Detailed breakdown from GED calculation
        
        Returns:
            Dictionary with component scores
        """
        scores = {}
        
        # Semantic similarity score (based on name costs)
        total_name_cost = breakdown.get('total_name_cost', 0)
        num_substitutions = breakdown.get('num_node_substitutions', 0)
        
        if num_substitutions > 0:
            avg_name_cost = total_name_cost / num_substitutions
            semantic_score = (1 - avg_name_cost) * 100
        else:
            semantic_score = 100.0  # No substitutions needed
        
        scores['semantic_score'] = max(0, min(100, semantic_score))
        
        # Structural similarity score (based on node counts)
        num_nodes_key = breakdown.get('num_nodes_key', 1)
        num_nodes_student = breakdown.get('num_nodes_student', 1)
        node_diff = breakdown.get('node_difference', 0)
        
        structural_score = (1 - node_diff / max(num_nodes_key, num_nodes_student)) * 100
        scores['structural_score'] = max(0, min(100, structural_score))
        
        # Relationship accuracy score (based on edge correctness)
        num_edges_key = breakdown.get('num_edges_key', 1)
        num_edges_student = breakdown.get('num_edges_student', 1)
        num_wrong_edges = breakdown.get('num_edge_wrong_type', 0)
        edge_diff = breakdown.get('edge_difference', 0)
        
        if num_edges_key > 0:
            # Penalize both wrong types and missing/extra edges
            relationship_errors = num_wrong_edges + edge_diff
            relationship_score = (1 - relationship_errors / max(num_edges_key, num_edges_student)) * 100
        else:
            relationship_score = 100.0
        
        scores['relationship_score'] = max(0, min(100, relationship_score))
        
        return scores
    
    def generate_detailed_score_report(self, ged_value: float, breakdown: Dict[str, Any],
                                      diagram_id: str = 'unknown') -> Dict[str, Any]:
        """
        Generate comprehensive score report
        
        Args:
            ged_value: Graph Edit Distance value
            breakdown: Detailed breakdown from GED calculation
            diagram_id: Student diagram identifier
        
        Returns:
            Complete score report dictionary
        """
        # Calculate overall score
        penalties = self.config.get('assessment', {}).get('penalties', {})
        max_ged = self.estimate_max_ged(
            breakdown['num_nodes_key'],
            breakdown['num_edges_key'],
            penalties
        )
        
        overall_score = self.calculate_similarity_score(ged_value, max_ged)
        
        # Calculate component scores
        component_scores = self.calculate_component_scores(breakdown)
        
        # Build report
        report = {
            'diagram_id': diagram_id,
            'overall_score': overall_score,
            'ged_value': ged_value,
            'max_possible_ged': max_ged,
            'normalized_ged': ged_value / max_ged if max_ged > 0 else ged_value,
            
            # Component scores
            'semantic_score': component_scores.get('semantic_score', 0),
            'structural_score': component_scores.get('structural_score', 0),
            'relationship_score': component_scores.get('relationship_score', 0),
            
            # Detailed metrics
            'details': {
                'nodes_key': breakdown['num_nodes_key'],
                'nodes_student': breakdown['num_nodes_student'],
                'nodes_matched': breakdown['num_nodes_key'] - breakdown['node_difference'],
                'nodes_missing': max(0, breakdown['num_nodes_key'] - breakdown['num_nodes_student']),
                'nodes_extra': max(0, breakdown['num_nodes_student'] - breakdown['num_nodes_key']),
                
                'edges_key': breakdown['num_edges_key'],
                'edges_student': breakdown['num_edges_student'],
                'edges_wrong_type': breakdown.get('num_edge_wrong_type', 0),
                'edges_missing': max(0, breakdown['num_edges_key'] - breakdown['num_edges_student']),
                'edges_extra': max(0, breakdown['num_edges_student'] - breakdown['num_edges_key']),
                
                'total_name_cost': breakdown.get('total_name_cost', 0),
                'total_attr_cost': breakdown.get('total_attr_cost', 0),
                'total_method_cost': breakdown.get('total_method_cost', 0),
            },
            
            # Edit operations
            'num_edit_operations': len(breakdown.get('edit_operations', [])),
            'edit_operations_summary': self._summarize_edit_operations(
                breakdown.get('edit_operations', [])
            )
        }
        
        return report
    
    def _summarize_edit_operations(self, edit_operations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Summarize edit operations
        
        Args:
            edit_operations: List of edit operation dictionaries
        
        Returns:
            Summary statistics
        """
        summary = {
            'node_substitutions': 0,
            'edge_wrong_types': 0,
            'total_operations': len(edit_operations)
        }
        
        for op in edit_operations:
            op_type = op.get('type', '')
            
            if op_type == 'node_substitution':
                summary['node_substitutions'] += 1
            elif op_type == 'edge_wrong_type':
                summary['edge_wrong_types'] += 1
        
        return summary
    
    def add_human_score(self, diagram_metadata: Dict[str, Any], 
                       base_score: float = None,
                       noise_std: float = 3.0) -> float:
        """
        Generate simulated human score based on variation type
        
        Args:
            diagram_metadata: Metadata including variation_type and expected_score_range
            base_score: Base score to add noise to (if None, use middle of expected range)
            noise_std: Standard deviation of noise to add
        
        Returns:
            Simulated human score
        """
        expected_range = diagram_metadata.get('expected_score_range', [0, 100])
        
        if base_score is None:
            # Use middle of expected range as base
            base_score = (expected_range[0] + expected_range[1]) / 2
        
        # Add noise
        noise = np.random.normal(0, noise_std)
        human_score = base_score + noise
        
        # Clamp to expected range (with some tolerance)
        min_score = max(0, expected_range[0] - 10)
        max_score = min(100, expected_range[1] + 10)
        human_score = np.clip(human_score, min_score, max_score)
        
        return float(human_score)
    
    def batch_calculate_scores(self, ged_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Calculate scores for multiple diagrams
        
        Args:
            ged_results: List of GED calculation results
        
        Returns:
            List of score reports
        """
        reports = []
        
        for result in ged_results:
            report = self.generate_detailed_score_report(
                result['ged_value'],
                result['breakdown'],
                result.get('diagram_id', 'unknown')
            )
            reports.append(report)
        
        logging.info(f"Generated {len(reports)} score reports")
        
        return reports
