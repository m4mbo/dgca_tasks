import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    mutex = parser.add_mutually_exclusive_group(required=True)

    # mga settings
    parser.add_argument('--pop_size', type=int, default=30, help='Name of dataset for trainig.')
    parser.add_argument('--mutate_rate', type=float, default=0.02, help='Mutation rate.')
    parser.add_argument('--cross_rate', type=float, default=0.5, help='Crossover rate.')
    parser.add_argument('--cross_style', type=str, default='cols', help='Crossover style.')
    parser.add_argument('--n_trials', type=int, default=5000, help='Number of trials for MGA.')
    
    # reservoir settings
    parser.add_argument('--input_nodes', type=int, default=0, help='Number reservoir input nodes (not to confuse with input units).')
    parser.add_argument('--output_nodes', type=int, default=0, help='Number reservoir output nodes (not to confuse with output units).')
    parser.add_argument('--max_size', type=int, help='Maximum reservoir size.')
    
    # task settings
    parser.add_argument('--order', type=int, default=10, help='Order for NARMA sequence.')
    mutex.add_argument('--task', type=str, default=None, help='Benchmark task name.')
    mutex.add_argument('--metric', type=str, default=None, help='Metric type (options: KR, GM, LMC or combined).')
    
    # extra settings
    parser.add_argument('--verbose', action='store_true', help='Enable verbose mode.')
    parser.add_argument('--output_file', type=str, default='fitness.parquet', help='Output file name.')

    return parser.parse_args()