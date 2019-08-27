import numpy as np
import random
import json


def breed(parent1, parent2):
    crossover1 = []
    crossover2 = []
    mutation_rate = 0.05

    gene1 = int(random.random() * len(parent1))
    gene2 = int(random.random() * len(parent2))
    start = min(gene1, gene2)
    end = max(gene1, gene2)

    for i in range(start, end):
        crossover1.append(parent1[i])

    for r in range(0, start):
        crossover2.append(parent2[r])
    for r in range(end, len(parent2)):
        crossover2.append(parent2[r])

    offspring = crossover1 + crossover2
    mutate(offspring, mutation_rate)

    return offspring


def mutate(offspring, mutation_rate):
    for gene in range(len(offspring)):
        mutation_prob = random.random()
        if mutation_prob < mutation_rate:
            offspring[gene] = random.uniform(0.001, 1.0)

    return offspring


def select(population):
    scores = []
    for sample in population:
        scores.append(sample[1])
    total = sum(scores)

    prob_selection_array = []
    for elem in scores:
        prob_selection_array.append(elem / total)

    return prob_selection_array


def mating(population, elite_size, pop_size):
    rates = []
    for elem in population:
        rates.append(elem[0])

    elites = []
    for i in range(elite_size):
        elites.append(rates[i])
    # print(elites)

    prob_selection_array = select(population)

    rates = []
    for elem in population:
        rates.append(elem[0])

    s1 = np.random.choice(pop_size,  p=prob_selection_array)
    s2 = np.random.choice(pop_size, p=prob_selection_array)

    parent1 = rates[s1]
    parent2 = rates[s2]

    new_pop = []
    for r in range(len(population)-elite_size):
        new_pop.append(breed(parent1, parent2))

    next_gen = new_pop + elites
    # print(next_gen)

    return next_gen


if __name__ == "__main__":
    num_rxns = 15
    pop_size = 40
    elite_size = int(0.15 * pop_size)
    fitness_comp = []

    def generate_rk(num_rxns):
        k_array = []
        for i in range(num_rxns):
            k_array.append(random.uniform(0.001, 1.0))
        return k_array

    next_gen = []
    for r in range(pop_size):
        next_gen.append(generate_rk(num_rxns))

    with open('simulation_data.txt', 'r') as f:
        population = json.loads(f.read())
    # print(population)

    def sort_fitness(val):
        return val[1]
    population.sort(key=sort_fitness)
    # print(population)

    fitness_comp = []
    for elem in population:
        fitness_comp.append(elem[1])

    fitness_comp.sort()
    # print(fitness_comp)

    best_fitness = fitness_comp[0]

    with open('fitness_data.txt', 'a') as f:
        f.write("%s\n" % str(best_fitness))

    next_gen = mating(population, elite_size, pop_size)

    with open('rate_constants.txt', 'w') as f:
        f.write(json.dumps(next_gen))

    with open('simulation_data.txt', 'w') as f:
        f.write(json.dumps(population))
