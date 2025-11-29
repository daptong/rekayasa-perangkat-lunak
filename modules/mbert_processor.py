"""
mBERT Processor Module
Handles multilingual BERT embeddings and semantic similarity calculations
"""

import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from typing import Dict, Tuple, Optional
import logging


class MBERTProcessor:
    """Process text using multilingual BERT for semantic embeddings"""
    
    def __init__(self, model_name: str = 'bert-base-multilingual-cased', 
                 max_length: int = 512,
                 device: Optional[str] = None):
        """
        Initialize mBERT model and tokenizer
        
        Args:
            model_name: HuggingFace model name
            max_length: Maximum sequence length
            device: 'cuda' or 'cpu', auto-detect if None
        """
        self.model_name = model_name
        self.max_length = max_length
        
        # Detect device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logging.info(f"Loading mBERT model: {model_name} on {self.device}")
        
        # Load model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Cache for embeddings to avoid redundant computation
        self.embedding_cache: Dict[str, np.ndarray] = {}
        
        logging.info("mBERT model loaded successfully")
    
    def generate_embedding(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        Generate mBERT embedding for text
        
        Args:
            text: Input string (English or Indonesian)
            use_cache: Whether to use cached embeddings
        
        Returns:
            768-dimensional vector representation
        """
        # Check cache
        if use_cache and text in self.embedding_cache:
            return self.embedding_cache[text]
        
        # Tokenize input
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=self.max_length
        )
        
        # Move to device
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Use [CLS] token embedding (first token)
        embedding = outputs.last_hidden_state[:, 0, :].squeeze()
        
        # Convert to numpy
        embedding_np = embedding.cpu().numpy()
        
        # Cache the result
        if use_cache:
            self.embedding_cache[text] = embedding_np
        
        return embedding_np
    
    def generate_mean_pooling_embedding(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        Generate embedding using mean pooling of all tokens
        
        Args:
            text: Input string
            use_cache: Whether to use cached embeddings
        
        Returns:
            768-dimensional vector representation
        """
        cache_key = f"mean_pool_{text}"
        
        # Check cache
        if use_cache and cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        # Tokenize input
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=self.max_length
        )
        
        # Move to device
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Mean pooling - take attention mask into account
        token_embeddings = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']
        
        # Expand attention mask
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        # Sum embeddings and divide by number of tokens
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embedding = (sum_embeddings / sum_mask).squeeze()
        
        # Convert to numpy
        embedding_np = embedding.cpu().numpy()
        
        # Cache the result
        if use_cache:
            self.embedding_cache[cache_key] = embedding_np
        
        return embedding_np
    
    def calculate_cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
        
        Returns:
            Cosine similarity value between -1 and 1
        """
        # Handle zero vectors
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        cosine_sim = np.dot(embedding1, embedding2) / (norm1 * norm2)
        
        return float(cosine_sim)
    
    def calculate_semantic_cost(self, label1: str, label2: str) -> float:
        """
        Calculate semantic cost between two labels
        
        Cost_sem(u, v) = 1 - CosineSimilarity(mBERT(u), mBERT(v))
        
        Args:
            label1: First label (English or Indonesian)
            label2: Second label (English or Indonesian)
        
        Returns:
            Semantic cost between 0 (identical meaning) and 2 (opposite meaning)
        """
        # Generate embeddings
        embedding1 = self.generate_embedding(label1)
        embedding2 = self.generate_embedding(label2)
        
        # Calculate cosine similarity
        cosine_sim = self.calculate_cosine_similarity(embedding1, embedding2)
        
        # Convert to cost (distance)
        semantic_cost = 1 - cosine_sim
        
        return semantic_cost
    
    def calculate_semantic_similarity(self, label1: str, label2: str) -> float:
        """
        Calculate semantic similarity (opposite of cost)
        
        Args:
            label1: First label
            label2: Second label
        
        Returns:
            Similarity score between -1 and 1
        """
        embedding1 = self.generate_embedding(label1)
        embedding2 = self.generate_embedding(label2)
        
        return self.calculate_cosine_similarity(embedding1, embedding2)
    
    def batch_generate_embeddings(self, texts: list) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for multiple texts in batch
        
        Args:
            texts: List of text strings
        
        Returns:
            Dictionary mapping text to embedding
        """
        embeddings = {}
        
        for text in texts:
            embeddings[text] = self.generate_embedding(text)
        
        return embeddings
    
    def clear_cache(self):
        """Clear the embedding cache"""
        self.embedding_cache.clear()
        logging.info("Embedding cache cleared")
    
    def get_cache_size(self) -> int:
        """Get number of cached embeddings"""
        return len(self.embedding_cache)
    
    def calculate_list_similarity(self, list1: list, list2: list) -> Tuple[float, int, int, int]:
        """
        Calculate similarity between two lists of items
        Uses Hungarian algorithm for optimal matching
        
        Args:
            list1: First list of text items
            list2: Second list of text items
        
        Returns:
            Tuple of (average_similarity, num_matched, num_missing, num_extra)
        """
        from scipy.optimize import linear_sum_assignment
        
        if not list1 and not list2:
            return 1.0, 0, 0, 0
        
        if not list1:
            return 0.0, 0, 0, len(list2)
        
        if not list2:
            return 0.0, 0, len(list1), 0
        
        # Create similarity matrix
        n1, n2 = len(list1), len(list2)
        similarity_matrix = np.zeros((n1, n2))
        
        for i, item1 in enumerate(list1):
            for j, item2 in enumerate(list2):
                similarity_matrix[i, j] = self.calculate_semantic_similarity(item1, item2)
        
        # Convert to cost matrix (maximize similarity = minimize cost)
        cost_matrix = 1 - similarity_matrix
        
        # Find optimal assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Calculate metrics
        matched = len(row_ind)
        total_similarity = sum(similarity_matrix[i, j] for i, j in zip(row_ind, col_ind))
        avg_similarity = total_similarity / max(n1, n2) if max(n1, n2) > 0 else 0
        
        missing = n1 - matched
        extra = n2 - matched
        
        return avg_similarity, matched, missing, extra
    
    def preload_common_embeddings(self, texts: list):
        """
        Preload embeddings for common texts
        
        Args:
            texts: List of texts to preload
        """
        logging.info(f"Preloading {len(texts)} embeddings...")
        for text in texts:
            self.generate_embedding(text)
        logging.info(f"Preloaded {len(texts)} embeddings into cache")


def calculate_attribute_similarity(attributes1: list, attributes2: list, 
                                   mbert_processor: MBERTProcessor) -> float:
    """
    Calculate similarity between two attribute lists
    
    Args:
        attributes1: List of attribute dictionaries from first class
        attributes2: List of attribute dictionaries from second class
        mbert_processor: MBERTProcessor instance
    
    Returns:
        Similarity cost (0 = identical, 1 = completely different)
    """
    if not attributes1 and not attributes2:
        return 0.0
    
    if not attributes1 or not attributes2:
        return 1.0
    
    # Create text representations of attributes
    attr_texts1 = [f"{attr['name']}: {attr['type']}" for attr in attributes1]
    attr_texts2 = [f"{attr['name']}: {attr['type']}" for attr in attributes2]
    
    # Calculate similarity
    similarity, matched, missing, extra = mbert_processor.calculate_list_similarity(
        attr_texts1, attr_texts2
    )
    
    # Convert to cost
    # Penalize missing/extra attributes
    penalty = (missing + extra) / max(len(attr_texts1), len(attr_texts2))
    cost = (1 - similarity) + penalty * 0.5
    
    return min(cost, 1.0)


def calculate_method_similarity(methods1: list, methods2: list, 
                                mbert_processor: MBERTProcessor) -> float:
    """
    Calculate similarity between two method lists
    
    Args:
        methods1: List of method dictionaries from first class
        methods2: List of method dictionaries from second class
        mbert_processor: MBERTProcessor instance
    
    Returns:
        Similarity cost (0 = identical, 1 = completely different)
    """
    if not methods1 and not methods2:
        return 0.0
    
    if not methods1 or not methods2:
        return 1.0
    
    # Create text representations of methods
    method_texts1 = []
    for method in methods1:
        params = ", ".join(method.get('parameters', []))
        method_texts1.append(f"{method['name']}({params}): {method.get('return_type', 'void')}")
    
    method_texts2 = []
    for method in methods2:
        params = ", ".join(method.get('parameters', []))
        method_texts2.append(f"{method['name']}({params}): {method.get('return_type', 'void')}")
    
    # Calculate similarity
    similarity, matched, missing, extra = mbert_processor.calculate_list_similarity(
        method_texts1, method_texts2
    )
    
    # Convert to cost
    penalty = (missing + extra) / max(len(method_texts1), len(method_texts2))
    cost = (1 - similarity) + penalty * 0.5
    
    return min(cost, 1.0)
