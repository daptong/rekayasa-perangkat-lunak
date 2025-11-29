"""
Graph Builder Module
Constructs NetworkX graphs from UML JSON diagrams with mBERT embeddings
"""

import networkx as nx
import logging
from typing import Dict, Any, Tuple
from modules.mbert_processor import MBERTProcessor


class GraphBuilder:
    """Build graph representations from UML diagram JSON"""
    
    def __init__(self, mbert_processor: MBERTProcessor):
        """
        Initialize graph builder
        
        Args:
            mbert_processor: MBERTProcessor instance for embeddings
        """
        self.mbert_processor = mbert_processor
    
    def build_graph_from_json(self, diagram_json: Dict[str, Any]) -> nx.DiGraph:
        """
        Construct graph G = (V, E, μ, ν) from JSON
        
        V = set of nodes (classes)
        E = set of edges (relationships)
        μ = node labeling function (mBERT embeddings)
        ν = edge labeling function (relationship types)
        
        Args:
            diagram_json: UML diagram in JSON format
        
        Returns:
            NetworkX DiGraph with node and edge attributes
        """
        G = nx.DiGraph()
        G.graph['diagram_id'] = diagram_json.get('diagram_id', 'unknown')
        G.graph['language'] = diagram_json.get('language', 'unknown')
        
        # Add nodes (classes)
        for cls in diagram_json['classes']:
            self._add_class_node(G, cls)
        
        # Add edges (relationships)
        for rel in diagram_json['relationships']:
            self._add_relationship_edge(G, rel)
        
        logging.info(f"Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        return G
    
    def _add_class_node(self, G: nx.DiGraph, cls: Dict[str, Any]):
        """
        Add a class as a node to the graph
        
        Args:
            G: NetworkX graph
            cls: Class dictionary from JSON
        """
        class_id = cls['id']
        class_name = cls['name']
        
        # Build composite label for the class
        # Format: "ClassName | attr1: type1, attr2: type2 | method1(), method2()"
        attributes_str = ", ".join([
            f"{attr['name']}: {attr['type']}" 
            for attr in cls.get('attributes', [])
        ])
        
        methods_str = ", ".join([
            f"{method['name']}({', '.join(method.get('parameters', []))}): {method.get('return_type', 'void')}"
            for method in cls.get('methods', [])
        ])
        
        full_label = f"{class_name}"
        if attributes_str:
            full_label += f" | {attributes_str}"
        if methods_str:
            full_label += f" | {methods_str}"
        
        # Generate mBERT embedding for the full label
        embedding = self.mbert_processor.generate_embedding(full_label)
        
        # Also generate separate embeddings for components
        name_embedding = self.mbert_processor.generate_embedding(class_name)
        
        # Add node with all attributes
        G.add_node(
            class_id,
            name=class_name,
            attributes=cls.get('attributes', []),
            methods=cls.get('methods', []),
            full_label=full_label,
            embedding=embedding,
            name_embedding=name_embedding
        )
    
    def _add_relationship_edge(self, G: nx.DiGraph, rel: Dict[str, Any]):
        """
        Add a relationship as an edge to the graph
        
        Args:
            G: NetworkX graph
            rel: Relationship dictionary from JSON
        """
        source = rel['source']
        target = rel['target']
        
        # Check if both nodes exist
        if source not in G.nodes() or target not in G.nodes():
            logging.warning(f"Skipping relationship {rel['id']}: source or target node not found")
            return
        
        # Add edge with attributes
        G.add_edge(
            source,
            target,
            rel_id=rel['id'],
            rel_type=rel['type'],
            label=rel.get('label', ''),
            multiplicity_source=rel.get('multiplicity_source', ''),
            multiplicity_target=rel.get('multiplicity_target', '')
        )
    
    def get_node_label(self, G: nx.DiGraph, node_id: str) -> str:
        """
        Get the full label of a node
        
        Args:
            G: NetworkX graph
            node_id: Node identifier
        
        Returns:
            Full label string
        """
        if node_id not in G.nodes():
            return ""
        
        return G.nodes[node_id].get('full_label', '')
    
    def get_node_embedding(self, G: nx.DiGraph, node_id: str):
        """
        Get the mBERT embedding of a node
        
        Args:
            G: NetworkX graph
            node_id: Node identifier
        
        Returns:
            Embedding vector or None
        """
        if node_id not in G.nodes():
            return None
        
        return G.nodes[node_id].get('embedding')
    
    def get_edge_type(self, G: nx.DiGraph, source: str, target: str) -> str:
        """
        Get the type of an edge
        
        Args:
            G: NetworkX graph
            source: Source node ID
            target: Target node ID
        
        Returns:
            Edge type string or empty string
        """
        if not G.has_edge(source, target):
            return ""
        
        return G[source][target].get('rel_type', '')
    
    def compare_graphs(self, G1: nx.DiGraph, G2: nx.DiGraph) -> Dict[str, Any]:
        """
        Compare two graphs and return basic statistics
        
        Args:
            G1: First graph
            G2: Second graph
        
        Returns:
            Dictionary with comparison statistics
        """
        stats = {
            'num_nodes_g1': G1.number_of_nodes(),
            'num_nodes_g2': G2.number_of_nodes(),
            'num_edges_g1': G1.number_of_edges(),
            'num_edges_g2': G2.number_of_edges(),
            'node_difference': abs(G1.number_of_nodes() - G2.number_of_nodes()),
            'edge_difference': abs(G1.number_of_edges() - G2.number_of_edges())
        }
        
        # Find common nodes by comparing labels semantically
        g1_labels = {node: G1.nodes[node]['name'] for node in G1.nodes()}
        g2_labels = {node: G2.nodes[node]['name'] for node in G2.nodes()}
        
        # Simple exact match for now (semantic matching done in GED)
        g1_names = set(g1_labels.values())
        g2_names = set(g2_labels.values())
        
        stats['common_node_names'] = len(g1_names & g2_names)
        stats['unique_to_g1'] = len(g1_names - g2_names)
        stats['unique_to_g2'] = len(g2_names - g1_names)
        
        return stats
    
    def visualize_graph(self, G: nx.DiGraph, output_file: str = None):
        """
        Create a simple text representation of the graph
        
        Args:
            G: NetworkX graph
            output_file: Optional file to save visualization
        """
        lines = []
        lines.append(f"Graph: {G.graph.get('diagram_id', 'unknown')}")
        lines.append(f"Language: {G.graph.get('language', 'unknown')}")
        lines.append(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
        lines.append("\nClasses:")
        
        for node in G.nodes():
            node_data = G.nodes[node]
            lines.append(f"  - {node}: {node_data['name']}")
            lines.append(f"    Attributes: {len(node_data.get('attributes', []))}")
            lines.append(f"    Methods: {len(node_data.get('methods', []))}")
        
        lines.append("\nRelationships:")
        for source, target, data in G.edges(data=True):
            source_name = G.nodes[source]['name']
            target_name = G.nodes[target]['name']
            rel_type = data.get('rel_type', 'unknown')
            lines.append(f"  - {source_name} --[{rel_type}]--> {target_name}")
        
        output = "\n".join(lines)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(output)
            logging.info(f"Graph visualization saved to {output_file}")
        
        return output
    
    def extract_all_labels(self, G: nx.DiGraph) -> Dict[str, list]:
        """
        Extract all unique labels from graph for embedding preloading
        
        Args:
            G: NetworkX graph
        
        Returns:
            Dictionary with different label types
        """
        labels = {
            'class_names': [],
            'full_labels': [],
            'attributes': [],
            'methods': []
        }
        
        for node in G.nodes():
            node_data = G.nodes[node]
            
            # Class name
            labels['class_names'].append(node_data['name'])
            
            # Full label
            labels['full_labels'].append(node_data['full_label'])
            
            # Attributes
            for attr in node_data.get('attributes', []):
                attr_text = f"{attr['name']}: {attr['type']}"
                labels['attributes'].append(attr_text)
            
            # Methods
            for method in node_data.get('methods', []):
                params = ", ".join(method.get('parameters', []))
                method_text = f"{method['name']}({params}): {method.get('return_type', 'void')}"
                labels['methods'].append(method_text)
        
        return labels


def build_graph_pair(key_diagram: Dict[str, Any], student_diagram: Dict[str, Any],
                     mbert_processor: MBERTProcessor) -> Tuple[nx.DiGraph, nx.DiGraph]:
    """
    Build graph pair for comparison
    
    Args:
        key_diagram: Key reference diagram JSON
        student_diagram: Student diagram JSON
        mbert_processor: MBERTProcessor instance
    
    Returns:
        Tuple of (key_graph, student_graph)
    """
    builder = GraphBuilder(mbert_processor)
    
    G_key = builder.build_graph_from_json(key_diagram)
    G_student = builder.build_graph_from_json(student_diagram)
    
    return G_key, G_student
