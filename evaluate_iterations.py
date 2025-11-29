"""
Aggregate Evaluation Across 5 Iterations
Reads per-iteration results, combines them, evaluates overall metrics,
and produces aggregate reports and visualizations.
"""

import os
import json
import yaml
import logging
import pandas as pd
import numpy as np
from typing import List

from modules.evaluator import Evaluator
from modules.visualizer import Visualizer


def setup_logging(log_file: str = 'output/aggregate/aggregate.log'):
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
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def find_iteration_folders(base_output_dir: str, expected: int = 5) -> List[str]:
    folders = []
    for i in range(1, expected + 1):
        folder = os.path.join(base_output_dir, f"iter_{i:02d}")
        if os.path.isdir(folder):
            folders.append(folder)
    return folders


def read_results_csv(iter_folder: str) -> pd.DataFrame:
    csv_path = os.path.join(iter_folder, 'results.csv')
    if not os.path.exists(csv_path):
        logging.warning(f"Missing results.csv in {iter_folder}")
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    df['iteration'] = os.path.basename(iter_folder)
    return df


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


def main():
    print("="*70)
    print("AGGREGATE EVALUATION FOR 5 ITERATIONS")
    print("="*70)

    # Setup
    setup_logging()
    config = load_config()
    base_output = config['output']['output_dir']

    # Collect iteration folders
    iters = find_iteration_folders(base_output, expected=5)
    if not iters:
        print("No iteration folders found under 'output/'. Run main.py first.")
        return

    # Read and combine
    dfs = []
    for folder in iters:
        df = read_results_csv(folder)
        if not df.empty:
            dfs.append(df)

    if not dfs:
        print("No results.csv files found to aggregate.")
        return

    combined = pd.concat(dfs, ignore_index=True)

    # Ensure required columns exist
    required_cols = ['similarity_score', 'human_score']
    missing = [c for c in required_cols if c not in combined.columns]
    if missing:
        print(f"Missing required columns in combined results: {missing}")
        return

    # Evaluate combined
    evaluator = Evaluator(config)
    summary = evaluator.generate_evaluation_summary(combined.copy())

    # Output locations
    aggregate_dir = os.path.join(base_output, 'aggregate')
    viz_dir = os.path.join(aggregate_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)

    # Save combined CSV
    combined_csv = os.path.join(aggregate_dir, 'combined_results.csv')
    combined.to_csv(combined_csv, index=False, encoding='utf-8')

    # Save evaluation summary JSON
    summary_json = os.path.join(aggregate_dir, 'combined_evaluation_summary.json')
    with open(summary_json, 'w', encoding='utf-8') as f:
        json.dump(to_serializable(summary), f, indent=2, ensure_ascii=False)

    # Save text summary and Excel, plus visuals
    visualizer = Visualizer(viz_dir)
    summary_txt = os.path.join(aggregate_dir, 'combined_summary.txt')
    _ = visualizer.generate_text_summary(summary, summary_txt)

    excel_path = os.path.join(aggregate_dir, 'combined_assessment_report.xlsx')
    visualizer.export_to_excel(combined, summary, excel_path)

    # Visuals
    visualizer.create_all_visualizations(combined, summary)

    # Console summary
    metrics = summary['overall_metrics']
    print("\nAggregate Metrics (All Iterations):")
    print(f"  Pearson r: {metrics['pearson_r']:.4f}")
    print(f"  p-value  : {metrics['pearson_p_value']:.6f}")
    print(f"  MAE      : {metrics['mae']:.2f}")
    print(f"  RMSE     : {metrics['rmse']:.2f}")
    print(f"  Target (0.76): {'✓ PASSED' if metrics['meets_target'] else '✗ NOT MET'}")
    print("\nOutputs:")
    print(f"  • Combined CSV: {combined_csv}")
    print(f"  • Summary JSON: {summary_json}")
    print(f"  • Summary TXT : {summary_txt}")
    print(f"  • Excel       : {excel_path}")
    print(f"  • Visuals     : {viz_dir}")
    print("="*70)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nAggregation interrupted by user.")
        logging.warning("Aggregation interrupted by user")
    except Exception as e:
        print(f"\nERROR: {e}")
        logging.error(f"Fatal error: {e}", exc_info=True)
        raise
