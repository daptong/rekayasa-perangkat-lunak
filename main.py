"""
Main Execution Pipeline
UML Diagram Assessment with mBERT and GED
"""

import os
import json
import yaml
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm

from modules.diagram_generator import DiagramGenerator
from modules.mbert_processor import MBERTProcessor
from modules.graph_builder import GraphBuilder
from modules.ged_calculator import GEDCalculator
from modules.scorer import Scorer
from modules.evaluator import Evaluator
from modules.visualizer import Visualizer


def setup_logging(log_file: str = 'output/assessment.log'):
    """Setup logging configuration"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )


def load_config(config_path: str = 'config.yaml') -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main execution pipeline"""
    
    print("="*70)
    print("UML DIAGRAM ASSESSMENT SYSTEM")
    print("Multilingual Assessment using mBERT and Graph Edit Distance")
    print("="*70)
    print()
    
    # Setup
    setup_logging()
    logging.info("Starting UML Diagram Assessment System")
    
    # Load configuration
    config = load_config()
    logging.info("Configuration loaded")
    
    # Prepare base directories
    base_data_dir = config['output']['data_dir']
    base_output_dir = config['output']['output_dir']
    os.makedirs(base_data_dir, exist_ok=True)
    os.makedirs(base_output_dir, exist_ok=True)

    # ========================================================================
    # STEP 2: Initialize mBERT (load once for all iterations)
    # ========================================================================
    print("\n[2/6] Initializing mBERT Model...")
    logging.info("="*70)
    logging.info("STEP 2: Initializing mBERT")
    logging.info("="*70)
    mbert_config = config['mbert']
    mbert_processor = MBERTProcessor(
        model_name=mbert_config['model_name'],
        max_length=mbert_config['max_length']
    )
    print(f"  ✓ Loaded {mbert_config['model_name']}")

    # Iterate runs 1..5
    total_iterations = 5
    for iteration in range(1, total_iterations + 1):
        print("\n" + "="*70)
        print(f"ITERATION {iteration}/{total_iterations}")
        print("="*70)

        # Create iteration-specific directories
        data_dir = os.path.join(base_data_dir, f"iter_{iteration:02d}")
        output_dir = os.path.join(base_output_dir, f"iter_{iteration:02d}")
        viz_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(viz_dir, exist_ok=True)

        # ====================================================================
        # STEP 1: Generate Diagrams (re-generate each iteration)
        # ====================================================================
        print("\n[1/6] Generating UML Diagrams...")
        logging.info("="*70)
        logging.info(f"ITER {iteration}: Generating UML Diagrams")
        logging.info("="*70)

        generator = DiagramGenerator(config)

        logging.info("Generating key reference diagram (English)...")
        key_diagram = generator.generate_key_reference_diagram()

        num_diagrams = config['diagram_generation']['num_student_diagrams']
        logging.info(f"Generating {num_diagrams} student diagram variations (Indonesian)...")
        student_diagrams = generator.generate_student_variations(key_diagram)

        # Save diagrams under iteration folder
        generator.save_diagrams(key_diagram, student_diagrams, data_dir)
        print(f"  ✓ Generated 1 key diagram and {len(student_diagrams)} variations -> {data_dir}")

        # ====================================================================
        # STEP 3: Build Graphs
        # ====================================================================
        print("\n[3/6] Building Graph Representations...")
        logging.info("="*70)
        logging.info(f"ITER {iteration}: Building Graph Representations")
        logging.info("="*70)

        graph_builder = GraphBuilder(mbert_processor)
        logging.info("Building key reference graph...")
        G_key = graph_builder.build_graph_from_json(key_diagram)
        print(f"  ✓ Key graph: {G_key.number_of_nodes()} nodes, {G_key.number_of_edges()} edges")

        # ====================================================================
        # STEP 4: Assess All Diagrams
        # ====================================================================
        print("\n[4/6] Assessing Student Diagrams...")
        logging.info("="*70)
        logging.info(f"ITER {iteration}: Assessing All Diagrams")
        logging.info("="*70)

        ged_calculator = GEDCalculator(mbert_processor, config)
        scorer = Scorer(config)
        assessment_results = []

        for student_diagram in tqdm(student_diagrams, desc=f"Assessing iter {iteration}"):
            diagram_id = student_diagram['diagram_id']

            # Build student graph
            G_student = graph_builder.build_graph_from_json(student_diagram)

            # Calculate GED
            try:
                ged_value, breakdown = ged_calculator.calculate_optimized_ged(G_key, G_student)
            except Exception as e:
                logging.error(f"Error calculating GED for {diagram_id}: {e}")
                continue

            # Generate score report
            score_report = scorer.generate_detailed_score_report(
                ged_value, breakdown, diagram_id
            )

            # Add metadata
            score_report['variation_type'] = student_diagram.get('variation_type', 'unknown')
            score_report['expected_score_range'] = student_diagram.get('expected_score_range', [0, 100])
            score_report['modifications'] = student_diagram.get('modifications', [])

            # Generate human score
            human_score = scorer.add_human_score(student_diagram, score_report['overall_score'])
            score_report['human_score'] = human_score

            assessment_results.append(score_report)

        print(f"  ✓ Assessed {len(assessment_results)} diagrams")
        logging.info(f"Assessment complete for {len(assessment_results)} diagrams (iter {iteration})")

        # ====================================================================
        # STEP 5: Evaluate Performance
        # ====================================================================
        print("\n[5/6] Evaluating System Performance...")
        logging.info("="*70)
        logging.info(f"ITER {iteration}: Evaluation and Validation")
        logging.info("="*70)

        results_df = pd.DataFrame(assessment_results)
        for key in ['nodes_key', 'nodes_student', 'nodes_matched', 'nodes_missing',
                    'nodes_extra', 'edges_key', 'edges_student', 'edges_wrong_type']:
            results_df[key] = results_df['details'].apply(lambda x: x.get(key, 0))
        results_df = results_df.rename(columns={'overall_score': 'similarity_score'})

        evaluator = Evaluator(config)
        evaluation_summary = evaluator.generate_evaluation_summary(results_df)

        metrics = evaluation_summary['overall_metrics']
        print(f"\n  Pearson Correlation: {metrics['pearson_r']:.4f}")
        print(f"  P-value: {metrics['pearson_p_value']:.6f}")
        print(f"  MAE: {metrics['mae']:.2f}")
        print(f"  RMSE: {metrics['rmse']:.2f}")
        print(f"  Target (0.76): {'✓ PASSED' if metrics['meets_target'] else '✗ NOT MET'}")

        # ====================================================================
        # STEP 6: Generate Visualizations and Reports (iteration-specific)
        # ====================================================================
        print("\n[6/6] Generating Visualizations and Reports...")
        logging.info("="*70)
        logging.info(f"ITER {iteration}: Visualization and Reporting")
        logging.info("="*70)

        visualizer = Visualizer(viz_dir)
        visualizer.create_all_visualizations(results_df, evaluation_summary)
        print(f"  ✓ Visualizations saved to {viz_dir}/")

        excel_path = os.path.join(output_dir, 'assessment_report.xlsx')
        visualizer.export_to_excel(results_df, evaluation_summary, excel_path)
        print(f"  ✓ Excel report saved to {excel_path}")

        csv_path = os.path.join(output_dir, 'results.csv')
        visualizer.export_to_csv(results_df, csv_path)
        print(f"  ✓ CSV results saved to {csv_path}")

        summary_path = os.path.join(output_dir, 'summary.txt')
        _ = visualizer.generate_text_summary(evaluation_summary, summary_path)
        print(f"  ✓ Text summary saved to {summary_path}")

        # Save evaluation summary as JSON (robust to numpy/pandas and circular refs)
        json_path = os.path.join(output_dir, 'evaluation_summary.json')

        def to_serializable(obj, _seen=None):
            """Recursively convert objects to JSON-serializable types with cycle protection."""
            if _seen is None:
                _seen = set()
            if obj is None or isinstance(obj, (str, int, float, bool)):
                return obj
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            try:
                import pandas as _pd
                if isinstance(obj, _pd.Series):
                    return obj.to_list()
            except Exception:
                pass
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            oid = id(obj)
            if oid in _seen:
                return f"<CircularRef:{type(obj).__name__}>"
            _seen.add(oid)
            if isinstance(obj, dict):
                return {to_serializable(k, _seen): to_serializable(v, _seen) for k, v in obj.items()}
            if isinstance(obj, (list, tuple, set)):
                return [to_serializable(x, _seen) for x in obj]
            return str(obj)

        clean_summary = to_serializable(evaluation_summary)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(clean_summary, f, indent=2, ensure_ascii=False)
        print(f"  ✓ Evaluation JSON saved to {json_path}")

        # Iteration summary
        print("\n" + "-"*70)
        print(f"ITERATION {iteration} COMPLETE")
        print(f"Data: {data_dir}")
        print(f"Output: {output_dir}")
        print("-"*70)

    # All iterations done
    print("\n" + "="*70)
    print("ALL ITERATIONS COMPLETE!")
    print("Results saved under per-iteration folders in 'data/' and 'output/'")
    print("="*70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAssessment interrupted by user.")
        logging.warning("Assessment interrupted by user")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        logging.error(f"Fatal error: {e}", exc_info=True)
        raise
