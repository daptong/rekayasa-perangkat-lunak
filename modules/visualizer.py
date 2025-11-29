"""
Visualizer Module
Creates plots, charts, and comprehensive reports
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import logging
import os


class Visualizer:
    """Generate visualizations and reports"""
    
    def __init__(self, output_dir: str = 'output/visualizations'):
        """
        Initialize visualizer
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        sns.set_style('whitegrid')
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 10
    
    def plot_scatter_with_regression(self, results_df: pd.DataFrame, 
                                     metrics: Dict[str, Any],
                                     save_path: str = None):
        """
        Create scatter plot of System Score vs Human Score with regression line
        
        Args:
            results_df: DataFrame with results
            metrics: Evaluation metrics dictionary
            save_path: Path to save the figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Scatter plot
        ax.scatter(results_df['human_score'], results_df['similarity_score'], 
                  alpha=0.6, s=100, edgecolors='black', linewidths=0.5)
        
        # Perfect agreement line
        ax.plot([0, 100], [0, 100], 'r--', label='Perfect Agreement', alpha=0.5)
        
        # Regression line
        z = np.polyfit(results_df['human_score'], results_df['similarity_score'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(0, 100, 100)
        ax.plot(x_line, p(x_line), 'b-', label=f'Regression Line', linewidth=2)
        
        # Labels and title
        ax.set_xlabel('Human Score', fontsize=14, fontweight='bold')
        ax.set_ylabel('System Score', fontsize=14, fontweight='bold')
        ax.set_title('System vs Human Scores\nUML Diagram Assessment', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Add correlation coefficient
        r = metrics['pearson_r']
        p_value = metrics['pearson_p_value']
        textstr = f'Pearson r = {r:.4f}\np-value = {p_value:.6f}\nn = {len(results_df)}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
               verticalalignment='top', bbox=props)
        
        ax.legend(loc='lower right', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'scatter_plot.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saved scatter plot to {save_path}")
        plt.close()
    
    def plot_score_distribution_by_type(self, results_df: pd.DataFrame, 
                                       save_path: str = None):
        """
        Create box plots showing score distribution by variation type
        
        Args:
            results_df: DataFrame with results
            save_path: Path to save the figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Sort variation types
        variation_order = sorted(results_df['variation_type'].unique())
        
        # System scores
        sns.boxplot(data=results_df, x='variation_type', y='similarity_score', 
                   order=variation_order, ax=ax1, palette='Set2')
        ax1.set_title('System Scores by Variation Type', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Variation Type', fontsize=12)
        ax1.set_ylabel('System Score', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Human scores
        sns.boxplot(data=results_df, x='variation_type', y='human_score', 
                   order=variation_order, ax=ax2, palette='Set3')
        ax2.set_title('Human Scores by Variation Type', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Variation Type', fontsize=12)
        ax2.set_ylabel('Human Score', fontsize=12)
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'scores_by_type.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saved box plots to {save_path}")
        plt.close()
    
    def plot_error_distribution(self, results_df: pd.DataFrame, 
                               save_path: str = None):
        """
        Create histogram of errors (system - human)
        
        Args:
            results_df: DataFrame with results
            save_path: Path to save the figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        errors = results_df['similarity_score'] - results_df['human_score']
        
        ax.hist(errors, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        ax.axvline(np.mean(errors), color='green', linestyle='-', linewidth=2, 
                  label=f'Mean Error = {np.mean(errors):.2f}')
        
        ax.set_xlabel('Error (System Score - Human Score)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Assessment Errors', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'error_distribution.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saved error distribution to {save_path}")
        plt.close()
    
    def plot_confusion_matrix_heatmap(self, confusion_matrix: pd.DataFrame, 
                                     save_path: str = None):
        """
        Create heatmap of confusion matrix
        
        Args:
            confusion_matrix: Confusion matrix DataFrame
            save_path: Path to save the figure
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='YlOrRd', 
                   cbar_kws={'label': 'Count'}, ax=ax, linewidths=0.5)
        
        ax.set_title('Confusion Matrix - Score Categories', fontsize=14, fontweight='bold')
        ax.set_ylabel('Human Category', fontsize=12)
        ax.set_xlabel('System Category', fontsize=12)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'confusion_matrix.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saved confusion matrix to {save_path}")
        plt.close()
    
    def plot_component_scores(self, results_df: pd.DataFrame, 
                             save_path: str = None):
        """
        Create plot showing semantic, structural, and relationship scores
        
        Args:
            results_df: DataFrame with component scores
            save_path: Path to save the figure
        """
        if 'semantic_score' not in results_df.columns:
            logging.warning("Component scores not found in results")
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(results_df))
        width = 0.25
        
        ax.bar(x - width, results_df['semantic_score'], width, label='Semantic', alpha=0.8)
        ax.bar(x, results_df['structural_score'], width, label='Structural', alpha=0.8)
        ax.bar(x + width, results_df['relationship_score'], width, label='Relationship', alpha=0.8)
        
        ax.set_xlabel('Diagram Index', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Component Scores Across Diagrams', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'component_scores.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saved component scores to {save_path}")
        plt.close()
    
    def plot_correlation_by_variation(self, variation_metrics: pd.DataFrame, 
                                     save_path: str = None):
        """
        Create bar chart of correlation by variation type
        
        Args:
            variation_metrics: DataFrame with metrics by variation type
            save_path: Path to save the figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        variation_metrics = variation_metrics.sort_values('pearson_r', ascending=False)
        
        bars = ax.barh(variation_metrics['variation_type'], variation_metrics['pearson_r'], 
                      color='steelblue', edgecolor='black')
        
        # Color bars based on value
        for i, (bar, r) in enumerate(zip(bars, variation_metrics['pearson_r'])):
            if r >= 0.8:
                bar.set_color('green')
            elif r >= 0.6:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        ax.axvline(0.76, color='red', linestyle='--', linewidth=2, label='Target (0.76)')
        ax.set_xlabel('Pearson Correlation Coefficient', fontsize=12)
        ax.set_ylabel('Variation Type', fontsize=12)
        ax.set_title('Correlation by Variation Type', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'correlation_by_type.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saved correlation by type to {save_path}")
        plt.close()
    
    def create_all_visualizations(self, results_df: pd.DataFrame, 
                                  evaluation_summary: Dict[str, Any]):
        """
        Create all standard visualizations
        
        Args:
            results_df: DataFrame with all results
            evaluation_summary: Complete evaluation summary
        """
        logging.info("Creating all visualizations...")
        
        # 1. Scatter plot with regression
        self.plot_scatter_with_regression(results_df, evaluation_summary['overall_metrics'])
        
        # 2. Score distribution by type
        self.plot_score_distribution_by_type(results_df)
        
        # 3. Error distribution
        self.plot_error_distribution(results_df)
        
        # 4. Confusion matrix
        confusion_matrix = pd.DataFrame(evaluation_summary['confusion_matrix'])
        self.plot_confusion_matrix_heatmap(confusion_matrix)
        
        # 5. Component scores
        self.plot_component_scores(results_df)
        
        # 6. Correlation by variation
        variation_metrics = pd.DataFrame(evaluation_summary['variation_metrics'])
        self.plot_correlation_by_variation(variation_metrics)
        
        logging.info(f"All visualizations saved to {self.output_dir}/")
    
    def export_to_excel(self, results_df: pd.DataFrame, 
                       evaluation_summary: Dict[str, Any],
                       output_path: str = None):
        """
        Export results to Excel with multiple sheets
        
        Args:
            results_df: DataFrame with all results
            evaluation_summary: Complete evaluation summary
            output_path: Path to save Excel file
        """
        if output_path is None:
            output_path = os.path.join(os.path.dirname(self.output_dir), 'assessment_report.xlsx')
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Sheet 1: Detailed Results
            results_df.to_excel(writer, sheet_name='Detailed Results', index=False)
            
            # Sheet 2: Summary Statistics
            overall_metrics = pd.DataFrame([evaluation_summary['overall_metrics']])
            overall_metrics.to_excel(writer, sheet_name='Summary Statistics', index=False)
            
            # Sheet 3: Variation Type Analysis
            variation_metrics = pd.DataFrame(evaluation_summary['variation_metrics'])
            variation_metrics.to_excel(writer, sheet_name='By Variation Type', index=False)
            
            # Sheet 4: Confusion Matrix
            confusion_matrix = pd.DataFrame(evaluation_summary['confusion_matrix'])
            confusion_matrix.to_excel(writer, sheet_name='Confusion Matrix')
            
            # Sheet 5: High Error Cases
            if evaluation_summary['high_error_cases']:
                high_errors = pd.DataFrame(evaluation_summary['high_error_cases'])
                high_errors.to_excel(writer, sheet_name='High Error Cases', index=False)
            
            # Sheet 6: Agreement Rates
            agreement_rates = pd.DataFrame([evaluation_summary['agreement_rates']])
            agreement_rates.to_excel(writer, sheet_name='Agreement Rates', index=False)
        
        logging.info(f"Exported report to {output_path}")
    
    def export_to_csv(self, results_df: pd.DataFrame, output_path: str = None):
        """
        Export results to CSV
        
        Args:
            results_df: DataFrame with all results
            output_path: Path to save CSV file
        """
        if output_path is None:
            output_path = os.path.join(os.path.dirname(self.output_dir), 'results.csv')
        
        results_df.to_csv(output_path, index=False, encoding='utf-8')
        logging.info(f"Exported results to {output_path}")
    
    def generate_text_summary(self, evaluation_summary: Dict[str, Any], 
                            output_path: str = None) -> str:
        """
        Generate text summary report
        
        Args:
            evaluation_summary: Complete evaluation summary
            output_path: Optional path to save text report
        
        Returns:
            Summary text
        """
        lines = []
        lines.append("="*70)
        lines.append("UML DIAGRAM ASSESSMENT - EVALUATION SUMMARY")
        lines.append("="*70)
        lines.append("")
        
        metrics = evaluation_summary['overall_metrics']
        
        lines.append("OVERALL PERFORMANCE")
        lines.append("-" * 70)
        lines.append(f"Pearson Correlation Coefficient: {metrics['pearson_r']:.4f}")
        lines.append(f"P-value: {metrics['pearson_p_value']:.6f}")
        lines.append(f"Spearman Correlation: {metrics['spearman_r']:.4f}")
        lines.append(f"Mean Absolute Error: {metrics['mae']:.2f} points")
        lines.append(f"Root Mean Square Error: {metrics['rmse']:.2f} points")
        lines.append(f"Mean Bias: {metrics['mean_bias']:+.2f} points")
        lines.append(f"Agreement within ±5: {metrics['agreement_within_5']:.1f}%")
        lines.append(f"Agreement within ±10: {metrics['agreement_within_10']:.1f}%")
        lines.append(f"Number of samples: {metrics['num_samples']}")
        lines.append("")
        
        lines.append("TARGET ACHIEVEMENT")
        lines.append("-" * 70)
        lines.append(f"Target correlation: {metrics['target_correlation']:.2f}")
        lines.append(f"Achieved: {metrics['pearson_r']:.4f}")
        lines.append(f"Status: {'✓ PASSED' if metrics['meets_target'] else '✗ NOT MET'}")
        lines.append("")
        
        lines.append("STATISTICAL SIGNIFICANCE")
        lines.append("-" * 70)
        sig = evaluation_summary['statistical_significance']
        lines.append(f"Is significant (α=0.05): {'Yes' if sig['is_significant'] else 'No'}")
        lines.append(f"P-value: {sig['p_value']:.6f}")
        lines.append("")
        
        lines.append("="*70)
        
        summary_text = "\n".join(lines)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(summary_text)
            logging.info(f"Saved text summary to {output_path}")
        
        return summary_text
