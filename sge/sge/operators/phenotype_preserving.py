"""
Module: phenotype_preserving.py
PSGE-PP (phenotype-preserving PSGE) and the re-mapping rate.

In PSGE every individual is re-mapped with the current PCFG when it is evaluated, so a grammar update
changes the phenotype of individuals whose genotype has not changed ("re-mapping drift").
PSGE-PP removes that drift: right after each PCFG update, every USED codon of every individual is moved
to the same relative position inside the interval of its rule in the new grammar. Mapping the
re-encoded genotype with the new PCFG then yields exactly the same derivation, so the grammar only
influences new genetic material (mutation, codons drawn when a genotype runs out).

The interval of a rule is computed exactly as grammar._recursive_mapping selects rules:
  - normal regime: rules in index order, thresholds round(cumulative probability, 3);
  - depth-limited regime (depth >= max_depth - shortest_path[0]): only the shortest-path rules,
    in shortest_path order, with probabilities renormalised over them (uniform if they sum to 0).
A codon c selects the first rule whose threshold t satisfies c <= t, so rule k owns (t_{k-1}, t_k]
(and [0, t_0] for the first rule). The regime is decided by the depth stored in the codon, which is
the depth at which it was last mapped; since the derivation is preserved, so is that depth.

A rule whose interval in the new grammar is empty (its probability rounds away) cannot be preserved;
the codon is left unchanged and the event is counted as a preservation failure.

Re-mapping rate: fraction of the population whose phenotype changes when its (unchanged) genotype is
mapped with the new PCFG. It is a pure measurement: it works on copies and saves/restores the global
NumPy RNG state, because mapping draws a random codon when a genotype runs out of codons.

References: Plan de investigación "ritmo de adaptación de la gramática, re-mapeo y estabilidad en PSGE"
(2026-11-08), RQ-C.
"""

import numpy as np

import sge.grammar as grammar


# Statistics of the last call to reencode_population (read by the engine's trajectory logger).
LAST_REENCODE_STATS = {"codons": 0, "moved": 0, "failures": 0, "individuals_with_failures": 0}


def _limited(nt, depth):
    shortest_path = grammar.get_shortest_path()[(nt, 'NT')]
    return depth >= (grammar.get_max_depth() - shortest_path[0])


def rule_thresholds(probs, nt, depth, cache=None):
    """Ordered list of (rule_index, threshold) used by the mapping for non-terminal `nt` at `depth`.
    Replicates the floating-point accumulation of grammar._recursive_mapping exactly.
    The result depends only on (nt, regime), so `cache` (a dict owned by the caller, one per PCFG)
    avoids recomputing it for every codon."""
    if cache is not None:
        key = (nt, _limited(nt, depth))
        hit = cache.get(key)
        if hit is None:
            hit = cache[key] = rule_thresholds(probs, nt, depth)
        return hit
    nt_index = grammar.get_index_of_non_terminal()[nt]
    rules = grammar.get_dict()[nt]
    out = []
    if _limited(nt, depth):
        shortest_path = grammar.get_shortest_path()[(nt, 'NT')]
        prob_non_recursive = 0.0
        for rule in shortest_path[1:]:
            index = rules.index(rule)
            prob_non_recursive += grammar.get_probability(probs, nt_index, index)
        prob_aux = 0.0
        for rule in shortest_path[1:]:
            index = rules.index(rule)
            if prob_non_recursive == 0.0:
                new_prob = 1.0 / len(shortest_path[1:])
            else:
                new_prob = grammar.get_probability(probs, nt_index, index) / prob_non_recursive
            prob_aux += new_prob
            out.append((index, round(prob_aux, 3)))
    else:
        prob_aux = 0.0
        for index in range(len(rules)):
            prob_aux += grammar.get_probability(probs, nt_index, index)
            out.append((index, round(prob_aux, 3)))
    return out


def select_rule(probs, nt, codon, depth, cache=None):
    """Rule chosen by the mapping for `codon` (None if no threshold covers it)."""
    for index, threshold in rule_thresholds(probs, nt, depth, cache):
        if codon <= threshold:
            return index
    return None


def rule_interval(probs, nt, rule_index, depth, cache=None):
    """(low, high, is_first) of the codon interval of `rule_index`: codons in (low, high] select it
    ([0, high] if is_first). Returns None if the rule cannot be selected in this regime."""
    thresholds = rule_thresholds(probs, nt, depth, cache)
    previous = None
    for index, threshold in thresholds:
        if index == rule_index:
            if previous is None:
                return 0.0, threshold, True
            return previous, threshold, False
        previous = threshold if previous is None else max(previous, threshold)
    return None


def reencode_codon(codon, rule_index, depth, nt, probs_old, probs_new, cache_old=None, cache_new=None):
    """New codon at the same relative position of the rule's interval in the new grammar.
    Returns (new_codon, ok). ok is False when the rule cannot be preserved (codon returned unchanged)."""
    old = rule_interval(probs_old, nt, rule_index, depth, cache_old)
    new = rule_interval(probs_new, nt, rule_index, depth, cache_new)
    if old is None or new is None:
        return codon, False
    lo_old, hi_old, _ = old
    lo_new, hi_new, first_new = new
    width_old = hi_old - lo_old
    u = 0.0 if width_old <= 0.0 else (codon - lo_old) / width_old
    u = min(max(u, 0.0), 1.0)
    candidate = lo_new + u * (hi_new - lo_new)
    if not first_new and candidate <= lo_new:
        candidate = float(np.nextafter(lo_new, 1.0))
    if candidate > hi_new:
        candidate = hi_new
    candidate = float(min(max(candidate, 0.0), 1.0))
    if select_rule(probs_new, nt, candidate, depth, cache_new) != rule_index:
        return codon, False
    return candidate, True


def reencode_individual(ind, probs_old, probs_new, cache_old=None, cache_new=None):
    """Re-encodes the used codons of `ind` in place (genotype entries are replaced, never mutated,
    because offspring may share entry objects with their parents). Returns (codons, moved, failures)."""
    codons = moved = failures = 0
    non_terminals = list(grammar.get_non_terminals())
    mapping_values = ind.get('mapping_values')
    for gene_index, gene in enumerate(ind['genotype']):
        nt = non_terminals[gene_index]
        if len(grammar.get_dict()[nt]) <= 1:
            continue
        used = len(gene) if mapping_values is None else min(mapping_values[gene_index], len(gene))
        for position in range(used):
            rule_index, codon, depth = gene[position]
            codons += 1
            new_codon, ok = reencode_codon(codon, rule_index, depth, nt, probs_old, probs_new,
                                           cache_old, cache_new)
            if not ok:
                failures += 1
                continue
            if new_codon != codon:
                moved += 1
                gene[position] = [rule_index, new_codon, depth]
    return codons, moved, failures


def reencode_population(population, probs_old, probs_new):
    """PSGE-PP step: re-encode every individual of the population after a PCFG update."""
    stats = {"codons": 0, "moved": 0, "failures": 0, "individuals_with_failures": 0}
    cache_old, cache_new = {}, {}
    for ind in population:
        c, m, f = reencode_individual(ind, probs_old, probs_new, cache_old, cache_new)
        stats["codons"] += c
        stats["moved"] += m
        stats["failures"] += f
        stats["individuals_with_failures"] += int(f > 0)
    LAST_REENCODE_STATS.clear()
    LAST_REENCODE_STATS.update(stats)
    return stats


def reset_reencode_stats():
    LAST_REENCODE_STATS.clear()
    LAST_REENCODE_STATS.update({"codons": 0, "moved": 0, "failures": 0, "individuals_with_failures": 0})


def map_copy(ind, probs):
    """Phenotype of `ind` mapped with `probs`, on a copy and without touching the global RNG."""
    genotype = [[list(entry) for entry in gene] for gene in ind['genotype']]
    positions = [0] * len(genotype)
    state = np.random.get_state()
    try:
        phenotype, _, _ = grammar.mapping(probs, genotype, positions)
    finally:
        np.random.set_state(state)
    return phenotype


def remap_rate(population, probs, sample=None):
    """Fraction of individuals whose stored phenotype differs from the phenotype of their unchanged
    genotype mapped with `probs`. With `sample` = n < len(population), only n individuals at evenly
    spaced positions of the list are mapped (deterministic: no random numbers are drawn)."""
    if not population:
        return 0.0
    if sample and sample < len(population):
        positions = np.linspace(0, len(population) - 1, int(sample)).round().astype(int)
        subset = [population[i] for i in sorted(set(positions.tolist()))]
    else:
        subset = population
    changed = sum(1 for ind in subset if map_copy(ind, probs) != ind.get('phenotype'))
    return changed / float(len(subset))
