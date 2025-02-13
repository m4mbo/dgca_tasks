import os
import sys
from grow.runner import Runner
from grow.reservoir import get_seed
from evolve.fitness import TaskFitness, MetricFitness
from evolve.mga import ChromosomalMGA, EvolvableDGCA
from prop.tasks import narma, santa_fe
from util.parser import parse_arguments
from util.consts import N_STATES


if __name__ == "__main__":
    args = parse_arguments()

    project_dir = os.path.abspath(os.path.dirname(sys.argv[0]))

    conditions = {'max_size': 100, 
                  'min_size': 50}
    
    if args.series:
        print(f"Running {args.series} experiment...")
        fitness_fn = TaskFitness(series=narma if args.series == "narma" else santa_fe,
                                 conditions=conditions, 
                                 verbose=args.verbose, 
                                 order=args.order,
                                 fixed_series=True)
    elif args.metric:
        print(f"Running {args.metric} experiment...")
        fitness_fn = MetricFitness(metric=args.metric,
                                   conditions=conditions, 
                                   verbose=args.verbose)
        

    reservoir = get_seed(args.input_nodes, args.output_nodes, N_STATES)

    model = EvolvableDGCA(n_states=reservoir.n_states)  
    runner = Runner(max_steps=100, max_size=300)
    mga = ChromosomalMGA(popsize=args.pop_size,
                        seed_graph=reservoir,
                        model=model,
                        runner=runner,
                        fitness_fn=fitness_fn,
                        mutate_rate=args.mutate_rate,
                        cross_rate=args.cross_rate,
                        cross_style=args.cross_style,
                        parquet_filename=args.output_file)
    
    mga.run(steps=args.n_trials)