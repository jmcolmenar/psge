import sys
import sge.grammar as grammar
import sge.logger as logger
from datetime import datetime
from tqdm import tqdm
import copy
import numpy as np
from sge.operators.recombination import crossover
from sge.operators.mutation import mutate, mutate_level, mutation_prob_mutation, reset_mutation_stats, MUTATION_STATS
from sge.operators.selection import tournament
from sge.operators.update import independent_update
from sge.operators.stability_aware_update import perform_grammar_update, reset_sa_rng, compute_pcfg_kl_divergence
from sge.operators.phenotype_preserving import remap_rate, reset_reencode_stats, LAST_REENCODE_STATS
from sge.stability import calculate_pcfg_entropy
from sge.parameters import (
    params,
    set_parameters,
    load_parameters
)


# Trayectoria compacta por generación (params['TRACK_TRAJECTORY']). Solo mide: no usa el generador
# aleatorio global (remap_rate guarda y restaura su estado), así que no cambia la búsqueda.
TRAJECTORY_FIELDS = ["gen", "best_fitness", "best_gen_fitness", "median_fitness", "invalid_frac",
                     "mean_depth", "mean_used_codons", "unique_pct", "entropy", "kl_update", "remap_rate",
                     "pp_codons", "pp_moved", "pp_failures", "mut_codons", "mut_rule_changes"]
_trajectory = {k: [] for k in TRAJECTORY_FIELDS}


def get_trajectory():
    """Copia de la trayectoria de la última ejecución: dict campo -> lista (una entrada por generación)."""
    return {k: list(v) for k, v in _trajectory.items()}


def _reset_trajectory():
    for v in _trajectory.values():
        v.clear()


def _record_generation(it, population, best, best_gen, pcfg_before, invalid_fitness):
    pcfg = grammar.get_pcfg()
    mask = grammar.get_mask()
    changed = not np.array_equal(pcfg, pcfg_before)
    fitness = np.array([i['fitness'] for i in population], dtype=float)
    row = {
        "gen": it,
        "best_fitness": float(best['fitness']),
        "best_gen_fitness": float(best_gen['fitness']),
        "median_fitness": float(np.median(fitness)),
        "invalid_frac": float(np.mean(fitness >= invalid_fitness)),
        "mean_depth": float(np.mean([i['tree_depth'] for i in population])),
        "mean_used_codons": float(np.mean([sum(i['mapping_values']) for i in population])),
        "unique_pct": 100.0 * len(set(i['phenotype'] for i in population)) / len(population),
        "entropy": float(calculate_pcfg_entropy(pcfg, mask)),
        "kl_update": float(compute_pcfg_kl_divergence(pcfg, pcfg_before, mask)) if changed else 0.0,
        "remap_rate": float(remap_rate(population, pcfg, params.get('REMAP_RATE_SAMPLE'))) if changed else 0.0,
        "pp_codons": int(LAST_REENCODE_STATS["codons"]),
        "pp_moved": int(LAST_REENCODE_STATS["moved"]),
        "pp_failures": int(LAST_REENCODE_STATS["failures"]),
        # acumulados hasta esta generación (la mutación de la generación it aún no ha ocurrido)
        "mut_codons": int(MUTATION_STATS["codon_mutations"]),
        "mut_rule_changes": int(MUTATION_STATS["rule_changes"]),
    }
    for k in TRAJECTORY_FIELDS:
        _trajectory[k].append(row[k])
    reset_reencode_stats()


def generate_random_individual():
    genotype = [[] for _ in grammar.get_non_terminals()]
    tree_depth = grammar.recursive_individual_creation(genotype, grammar.start_rule()[0], 0, grammar.get_pcfg())
    if params['ADAPTIVE_MUTATION']:
        return {'genotype': genotype, 'fitness': None, 'tree_depth' : tree_depth, 'mutation_probs': [params['PROB_MUTATION'] for _ in genotype] }
    else:
        return {'genotype': genotype, 'fitness': None, 'tree_depth' : tree_depth}


def make_initial_population():
    for i in range(params['POPSIZE']):
        yield generate_random_individual()


def evaluate(ind, eval_func):
    mapping_values = [0 for _ in ind['genotype']]
    phen, tree_depth, grammar_counter = grammar.mapping(grammar.get_pcfg(), ind['genotype'], mapping_values)
    quality, other_info = eval_func.evaluate(phen)
    ind['phenotype'] = phen
    ind['fitness'] = quality
    ind['other_info'] = other_info
    ind['mapping_values'] = mapping_values
    ind['tree_depth'] = tree_depth
    ind['grammar_counter'] = grammar_counter


def setup(parameters_file_path = None, options_dict = None):
    if parameters_file_path is not None:
        load_parameters(file_name=parameters_file_path)
    if options_dict is not None:
        params.update(options_dict)
    else:
        set_parameters(sys.argv[1:])
    if params['SEED'] is None:
        params['SEED'] = int(datetime.now().microsecond)
    lf_suffix = "/" + str(params['LEARNING_FACTOR'] * 100)
    if not params['EXPERIMENT_NAME'].endswith(lf_suffix):
        params['EXPERIMENT_NAME'] += lf_suffix

    logger.prepare_dumps()
    np.random.seed(int(params['SEED']))
    reset_sa_rng(int(params['SEED']))  # SA-PSGE views use their own stream
    reset_mutation_stats()
    reset_reencode_stats()
    _reset_trajectory()
    grammar.set_path(params['GRAMMAR'])
    grammar.set_max_tree_depth(params['MAX_TREE_DEPTH'])
    grammar.set_min_init_tree_depth(params['MIN_TREE_DEPTH'])
    grammar.read_grammar()


def evolutionary_algorithm(evaluation_function=None, parameters_file=None, options_dict=None):
    setup(parameters_file_path=parameters_file, options_dict=options_dict)
    population = list(make_initial_population())
    flag = False    # alternate False - best overall
    best = None
    it = 0
    for i in tqdm(population):
        if i['fitness'] is None:
            evaluate(i, evaluation_function)
    while it <= params['GENERATIONS']:        

        population.sort(key=lambda x: x['fitness'])
        # best individual overall
        if not best:
            best = copy.deepcopy(population[0])
            best_gen = copy.deepcopy(best)
        elif population[0]['fitness'] <= best['fitness']:
            best = copy.deepcopy(population[0])
     
        track = params.get('TRACK_TRAJECTORY', False)
        pcfg_before = copy.deepcopy(grammar.get_pcfg()) if track else None
        perform_grammar_update(population, best_gen, flag, evaluation_function, it)
        flag = not flag
        if track:
            _record_generation(it, population, best, best_gen, pcfg_before,
                               params.get('INVALID_FITNESS', 9999999))

        if params['ADAPTIVE_LF']:
            params['LEARNING_FACTOR'] += params['ADAPTIVE_INCREMENT']

        logger.evolution_progress(it, population, best, best_gen, grammar.get_pcfg())

        new_population = []
        while len(new_population) < params['POPSIZE'] - params['ELITISM']:
            if np.random.uniform() < params['PROB_CROSSOVER']:
                p1 = tournament(population, params['TSIZE'])
                p2 = tournament(population, params['TSIZE'])
                ni = crossover(p1, p2)
            else:
                ni = tournament(population, params['TSIZE'])
            if params['ADAPTIVE_MUTATION']:
                # if we want to use Adaptive Facilitated Mutation
                ni = mutation_prob_mutation(ni)
                ni = mutate_level(ni)
            else:
                ni = mutate(ni, params['PROB_MUTATION'])                
            new_population.append(ni)

        for i in tqdm(new_population):
            evaluate(i, evaluation_function)
        new_population.sort(key=lambda x: x['fitness'])
        # best individual from the current generation
        best_gen = copy.deepcopy(new_population[0])

        if params['REMAP']:
            for i in tqdm(population[:params['ELITISM']]):
                evaluate(i, evaluation_function)
        new_population += population[:params['ELITISM']]

        population = new_population
        it += 1
    return best, copy.deepcopy(grammar.get_pcfg())


