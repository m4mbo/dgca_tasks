# 150 GA runs in parallel

from joblib import Parallel, delayed
from types import SimpleNamespace
from grow.runner import Runner
from grow.reservoir import get_seed
from evolve.fitness import TaskFitness, MetricFitness
from evolve.mga import ChromosomalMGA, EvolvableDGCA
from measure.tasks import narmax, santa_fe
from util.consts import N_STATES


def run_ga(run_id, args):
    """
    Runs a single GA experiment instance.
    """
    print(f"Starting GA run {run_id}...")

    conditions = {'max_size': args.max_size, 'min_size': 20}

    if args.task:
        fitness_fn = TaskFitness(series=narmax if args.task == "narma" else santa_fe,
                                 conditions=conditions, 
                                 verbose=False,
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
                        run_id=run_id,  
                        n_trials=args.n_trials,
                        cross_style=args.cross_style,
                        db_file=args.output_file)
    
    mga.run()
    print(f"Completed GA run {run_id}.")

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
        "output_file": "fitness.db",
        "num_jobs": 150
    }

    args = SimpleNamespace(**args_dict)

    num_parallel_jobs = 20  # match with cpu cores
    total_runs = 150 

    Parallel(n_jobs=num_parallel_jobs)(
        delayed(run_ga)(run_id, args) for run_id in range(total_runs)
    )

    print("All GA runs completed.")