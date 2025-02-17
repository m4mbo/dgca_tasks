# 150 GA runs in parallel

from joblib import Parallel, delayed
from types import SimpleNamespace
from grow.runner import Runner
from grow.reservoir import get_seed
from evolve.fitness import TaskFitness, MetricFitness
from evolve.mga import ChromosomalMGA, EvolvableDGCA
from measure.tasks import narmax, santa_fe
from util.consts import N_STATES


def run_ga(exp_id, args):
    """
    Runs a single GA experiment instance.
    """
    print(f"Starting GA run {exp_id}...")

    conditions = {'max_size': args.max_size, 'min_size': 20}

    if args.task:
        fitness_fn = TaskFitness(series=narmax if args.task == "narma" else santa_fe,
                                 conditions=conditions, 
                                 verbose=False,  # Assuming verbose is False if not provided
                                 order=args.order,
                                 fixed_series=True)
    elif args.metric:
        fitness_fn = MetricFitness(metric=args.metric,
                                   conditions=conditions, 
                                   verbose=False)

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
                        exp_id=exp_id,  
                        cross_style=args.cross_style,
                        parquet_file=args.output_file)
    
    mga.run(steps=args.n_trials)
    print(f"Completed GA run {exp_id}.")


if __name__ == "__main__":

    args_dict = {
        "pop_size": 30,
        "mutate_rate": 0.02,
        "cross_rate": 0.5,
        "cross_style": "cols",
        "n_trials": 2000,
        "input_nodes": 0,
        "output_nodes": 0,
        "order": 10,
        "task": "narma",
        "max_size": 100,
        "metric": None, 
        "output_file": "fitness.parquet",
        "num_jobs": 150
    }

    args = SimpleNamespace(**args_dict)

    num_parallel_jobs = 24  # match with cpu cores
    total_runs = 150 

    Parallel(n_jobs=num_parallel_jobs)(
        delayed(run_ga)(exp_id, args) for exp_id in range(total_runs)
    )

    print("All GA runs completed.")