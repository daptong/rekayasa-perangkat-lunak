"""
Evaluator Module
Performs Pearson correlation analysis and statistical validation
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging
from typing import Dict, Any, List, Tuple


class Evaluator:
    """Evaluate system performance against human scores"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize evaluator
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.correlation_threshold = config.get('evaluation', {}).get('correlation_threshold', 0.76)
    
    def evaluate_system_performance(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate Pearson correlation and other metrics
        
        Args:
            results_df: DataFrame with 'similarity_score' and 'human_score' columns
        
        Returns:
            Dictionary with evaluation metrics
        """
        system_scores = results_df['similarity_score'].values
        human_scores = results_df['human_score'].values
        
        # Calculate Pearson correlation
        r_pearson, p_value_pearson = pearsonr(system_scores, human_scores)
        
        # Calculate Spearman correlation (rank-based)
        r_spearman, p_value_spearman = spearmanr(system_scores, human_scores)
        
        # Calculate error metrics
        mae = mean_absolute_error(human_scores, system_scores)
        rmse = np.sqrt(mean_squared_error(human_scores, system_scores))
        
        # Calculate mean bias (system tendency to over/under-score)
        mean_bias = np.mean(system_scores - human_scores)
        
        # Calculate percentage of scores within tolerance
        tolerance_5 = np.sum(np.abs(system_scores - human_scores) <= 5) / len(system_scores) * 100
        tolerance_10 = np.sum(np.abs(system_scores - human_scores) <= 10) / len(system_scores) * 100
        
        metrics = {
            'pearson_r': r_pearson,
            'pearson_p_value': p_value_pearson,
            'spearman_r': r_spearman,
            'spearman_p_value': p_value_spearman,
            'mae': mae,
            'rmse': rmse,
            'mean_bias': mean_bias,
            'agreement_within_5': tolerance_5,
            'agreement_within_10': tolerance_10,
            'num_samples': len(system_scores),
            'target_correlation': self.correlation_threshold,
            'meets_target': r_pearson >= self.correlation_threshold
        }
        
        # Log results
        logging.info("="*60)
        logging.info("EVALUATION RESULTS")
        logging.info("="*60)
        logging.info(f"Pearson Correlation Coefficient (r): {r_pearson:.4f}")
        logging.info(f"P-value: {p_value_pearson:.6f}")
        logging.info(f"Spearman Correlation Coefficient: {r_spearman:.4f}")
        logging.info(f"Mean Absolute Error: {mae:.2f}")
        logging.info(f"Root Mean Square Error: {rmse:.2f}")
        logging.info(f"Mean Bias: {mean_bias:+.2f}")
        logging.info(f"Agreement within ±5 points: {tolerance_5:.1f}%")
        logging.info(f"Agreement within ±10 points: {tolerance_10:.1f}%")
        logging.info(f"Target correlation: {self.correlation_threshold:.2f}")
        logging.info(f"Meets target: {'YES' if metrics['meets_target'] else 'NO'}")
        logging.info("="*60)
        
        return metrics
    
    def evaluate_by_variation_type(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluate performance separately for each variation type
        
        Args:
            results_df: DataFrame with results
        
        Returns:
            DataFrame with metrics by variation type
        """
        variation_types = results_df['variation_type'].unique()
        
        metrics_by_type = []
        
        for var_type in variation_types:
            subset = results_df[results_df['variation_type'] == var_type]
            
            if len(subset) < 2:
                continue
            
            system_scores = subset['similarity_score'].values
            human_scores = subset['human_score'].values
            
            # Calculate metrics
            try:
                r_pearson, p_value = pearsonr(system_scores, human_scores)
            except:
                r_pearson, p_value = 0, 1.0
            
            mae = mean_absolute_error(human_scores, system_scores)
            rmse = np.sqrt(mean_squared_error(human_scores, system_scores))
            mean_bias = np.mean(system_scores - human_scores)
            
            metrics_by_type.append({
                'variation_type': var_type,
                'num_samples': len(subset),
                'pearson_r': r_pearson,
                'p_value': p_value,
                'mae': mae,
                'rmse': rmse,
                'mean_bias': mean_bias,
                'mean_system_score': np.mean(system_scores),
                'mean_human_score': np.mean(human_scores)
            })
        
        metrics_df = pd.DataFrame(metrics_by_type)
        metrics_df = metrics_df.sort_values('variation_type')
        
        logging.info("\nPerformance by Variation Type:")
        logging.info(metrics_df.to_string(index=False))
        
        return metrics_df
    
    def calculate_score_categories(self, score: float) -> str:
        """
        Categorize score into quality levels
        
        Args:
            score: Similarity score (0-100)
        
        Returns:
            Category string
        """
        if score >= 90:
            return 'Excellent'
        elif score >= 70:
            return 'Good'
        elif score >= 50:
            return 'Fair'
        else:
            return 'Poor'
    
    def generate_confusion_matrix(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate confusion matrix for score categories
        
        Args:
            results_df: DataFrame with results
        
        Returns:
            Confusion matrix as DataFrame
        """
        # Categorize scores
        results_df['system_category'] = results_df['similarity_score'].apply(
            self.calculate_score_categories
        )
        results_df['human_category'] = results_df['human_score'].apply(
            self.calculate_score_categories
        )
        
        # Create confusion matrix
        categories = ['Poor', 'Fair', 'Good', 'Excellent']
        confusion = pd.crosstab(
            results_df['human_category'],
            results_df['system_category'],
            rownames=['Human'],
            colnames=['System']
        )
        
        # Ensure all categories are present
        for cat in categories:
            if cat not in confusion.index:
                confusion.loc[cat] = 0
            if cat not in confusion.columns:
                confusion[cat] = 0
        
        # Reorder
        confusion = confusion.reindex(categories, axis=0, fill_value=0)
        confusion = confusion.reindex(categories, axis=1, fill_value=0)
        
        logging.info("\nConfusion Matrix (Score Categories):")
        logging.info(confusion)
        
        return confusion
    
    def analyze_errors(self, results_df: pd.DataFrame, threshold: float = 10.0) -> pd.DataFrame:
        """
        Analyze cases with large errors
        
        Args:
            results_df: DataFrame with results
            threshold: Error threshold for analysis
        
        Returns:
            DataFrame with high-error cases
        """
        results_df['error'] = results_df['similarity_score'] - results_df['human_score']
        results_df['abs_error'] = np.abs(results_df['error'])
        
        high_error_cases = results_df[results_df['abs_error'] > threshold].copy()
        high_error_cases = high_error_cases.sort_values('abs_error', ascending=False)
        
        if len(high_error_cases) > 0:
            logging.warning(f"\n{len(high_error_cases)} cases with error > {threshold}:")
            logging.warning(high_error_cases[[
                'diagram_id', 'variation_type', 'similarity_score', 'human_score', 'error'
            ]].to_string(index=False))
        else:
            logging.info(f"\nNo cases with error > {threshold}")
        
        return high_error_cases
    
    def calculate_agreement_rate(self, results_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate agreement rates at different tolerance levels
        
        Args:
            results_df: DataFrame with results
        
        Returns:
            Dictionary with agreement rates
        """
        results_df['error'] = np.abs(results_df['similarity_score'] - results_df['human_score'])
        
        agreement_rates = {}
        
        for tolerance in [3, 5, 7, 10, 15]:
            agreement = (results_df['error'] <= tolerance).sum() / len(results_df) * 100
            agreement_rates[f'within_{tolerance}'] = agreement
        
        logging.info("\nAgreement Rates:")
        for tolerance, rate in agreement_rates.items():
            logging.info(f"  Within ±{tolerance.split('_')[1]} points: {rate:.1f}%")
        
        return agreement_rates
    
    def generate_evaluation_summary(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation summary
        
        Args:
            results_df: DataFrame with all results
        
        Returns:
            Complete evaluation summary
        """
        # Overall metrics
        overall_metrics = self.evaluate_system_performance(results_df)
        
        # By variation type
        variation_metrics = self.evaluate_by_variation_type(results_df)
        
        # Confusion matrix
        confusion_matrix = self.generate_confusion_matrix(results_df)
        
        # Error analysis
        high_errors = self.analyze_errors(results_df)
        
        # Agreement rates
        agreement_rates = self.calculate_agreement_rate(results_df)
        
        # Calculate statistical significance
        is_significant = overall_metrics['pearson_p_value'] < 0.05
        
        summary = {
            'overall_metrics': overall_metrics,
            'variation_metrics': variation_metrics.to_dict('records'),
            'confusion_matrix': confusion_matrix.to_dict(),
            'high_error_cases': high_errors.to_dict('records') if len(high_errors) > 0 else [],
            'agreement_rates': agreement_rates,
            'statistical_significance': {
                'is_significant': is_significant,
                'significance_level': 0.05,
                'p_value': overall_metrics['pearson_p_value']
            }
        }
        
        return summary
    
    def compare_to_baseline(self, results_df: pd.DataFrame, 
                           baseline_method: str = 'mean') -> Dict[str, Any]:
        """
        Compare system performance to baseline methods
        
        Args:
            results_df: DataFrame with results
            baseline_method: 'mean', 'median', or 'random'
        
        Returns:
            Comparison results
        """
        human_scores = results_df['human_score'].values
        system_scores = results_df['similarity_score'].values
        
        # Generate baseline predictions
        if baseline_method == 'mean':
            baseline_predictions = np.full_like(human_scores, np.mean(human_scores))
        elif baseline_method == 'median':
            baseline_predictions = np.full_like(human_scores, np.median(human_scores))
        elif baseline_method == 'random':
            np.random.seed(42)
            baseline_predictions = np.random.uniform(0, 100, len(human_scores))
        else:
            raise ValueError(f"Unknown baseline method: {baseline_method}")
        
        # Calculate metrics for both
        system_mae = mean_absolute_error(human_scores, system_scores)
        baseline_mae = mean_absolute_error(human_scores, baseline_predictions)
        
        system_rmse = np.sqrt(mean_squared_error(human_scores, system_scores))
        baseline_rmse = np.sqrt(mean_squared_error(human_scores, baseline_predictions))
        
        system_r, _ = pearsonr(system_scores, human_scores)
        
        try:
            baseline_r, _ = pearsonr(baseline_predictions, human_scores)
        except:
            baseline_r = 0.0
        
        comparison = {
            'baseline_method': baseline_method,
            'system': {
                'pearson_r': system_r,
                'mae': system_mae,
                'rmse': system_rmse
            },
            'baseline': {
                'pearson_r': baseline_r,
                'mae': baseline_mae,
                'rmse': baseline_rmse
            },
            'improvement': {
                'pearson_r': system_r - baseline_r,
                'mae': baseline_mae - system_mae,
                'rmse': baseline_rmse - system_rmse
            }
        }
        
        logging.info(f"\nComparison to {baseline_method} baseline:")
        logging.info(f"  System Pearson r: {system_r:.4f} vs Baseline: {baseline_r:.4f}")
        logging.info(f"  System MAE: {system_mae:.2f} vs Baseline: {baseline_mae:.2f}")
        logging.info(f"  System RMSE: {system_rmse:.2f} vs Baseline: {baseline_rmse:.2f}")
        
        return comparison
