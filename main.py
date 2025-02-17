import os
import sys
from grow.runner import Runner
from grow.reservoir import get_seed
from evolve.fitness import TaskFitness, MetricFitness
from evolve.mga import ChromosomalMGA, EvolvableDGCA
from measure.tasks import narmax, santa_fe
from util.parser import parse_arguments
from util.consts import N_STATES


if __name__ == "__main__":
    args = parse_arguments()

    project_dir = os.path.abspath(os.path.dirname(sys.argv[0]))

    conditions = {'max_size': args.max_size, 
                  'min_size': 20}
    
    if args.task:
        print(f"Running {args.task} experiment...")
        fitness_fn = TaskFitness(series=narmax if args.task == "narma" else santa_fe,
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

    model = EvolvableDGCA(n_states=reservoir.n_states, hidden_size=None)  
    runner = Runner(max_steps=100, max_size=300)
    mga = ChromosomalMGA(popsize=args.pop_size,
                        seed_graph=reservoir,
                        model=model,
                        runner=runner,
                        fitness_fn=fitness_fn,
                        mutate_rate=args.mutate_rate,
                        cross_rate=args.cross_rate,
                        exp_id=args.exp_id,
                        cross_style=args.cross_style,
                        parquet_file=args.output_file)
    
    mga.run(steps=args.n_trials)