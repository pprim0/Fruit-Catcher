import random
import os
import numpy as np

def create_individual(individual_size):
    return [random.uniform(-1, 1) for _ in range(individual_size)]

def generate_population(individual_size, population_size):
    return [create_individual(individual_size) for _ in range(population_size)]

def evaluate_population(population, fitness_function, generation):
    seed = generation * 42  # mesma seed por geração para consistência
    return [(ind, fitness_function(ind, seed=seed)) for ind in population]

def select_elites(evaluated_population, elite_rate):
    sorted_population = sorted(evaluated_population, key=lambda x: x[1], reverse=True)
    elite_count = max(1, int(len(sorted_population) * elite_rate))
    return [ind for ind, _ in sorted_population[:elite_count]]

def crossover(parent1, parent2):
    alpha = random.random()
    return [alpha * a + (1 - alpha) * b for a, b in zip(parent1, parent2)]

def mutate(individual, mutation_rate, strength=1.0):
    return [
        gene + random.gauss(0, 0.3 * strength) if random.random() < mutation_rate else gene
        for gene in individual
    ]

def genetic_algorithm(individual_size, population_size, fitness_function, target_fitness, generations, elite_rate=0.1, mutation_rate=0.15):
    population = generate_population(individual_size, population_size)

    aux_file = "melhores_por_geracao.txt"
    open(aux_file, "w").close()  # limpa antes

    for gen in range(generations):
        seed = gen * 42
        evaluated = [(ind, fitness_function(ind, seed=seed)) for ind in population]
        evaluated.sort(key=lambda x: x[1], reverse=True)

        best_of_gen = evaluated[0][0]

        with open(aux_file, "a") as f:
            f.write(",".join(map(str, best_of_gen)) + "\n")

        fits = [fit for _, fit in evaluated]
        print(f"[Geração {gen+1}] Melhor: {fits[0]:.2f}, Média: {sum(fits)/len(fits):.2f}")

        elites = select_elites(evaluated, elite_rate)
        new_population = elites.copy()

        while len(new_population) < population_size:
            parents = random.sample(elites, 2)
            child = crossover(parents[0], parents[1])
            strength = 1.0 if len(new_population) < population_size * 0.8 else 2.0
            child = mutate(child, mutation_rate, strength)
            new_population.append(child)

        num_new = int(0.1 * population_size)
        new_population[-num_new:] = generate_population(individual_size, num_new)
        population = new_population[:population_size]

    # Avaliação final: média e desvio padrão
    def avg_and_std_fitness(ind):
        scores = [fitness_function(ind, seed=1234 + i) for i in range(10)]
        return np.mean(scores), np.std(scores)

    with open(aux_file, "r") as f:
        top_individuals = [list(map(float, line.strip().split(','))) for line in f]

    final_evals = []
    for ind in top_individuals:
        mean, std = avg_and_std_fitness(ind)
        final_evals.append((ind, mean, std))

    final_evals.sort(key=lambda x: (-x[1], x[2]))  # média desc, desvio asc
    best_final_ind, best_mean, best_std = final_evals[0]

    # Guardar top 5 no log
    with open("log_resultados.txt", "w") as logf:
        logf.write("Top 5 indivíduos finais:\n")
        for i, (_, mean, std) in enumerate(final_evals[:5]):
            logf.write(f"#{i+1}: Média = {mean:.2f}, Desvio = {std:.2f}\n")

    # Guardar o melhor final
    with open("best_individual.txt", "w") as f:
        f.write(",".join(map(str, best_final_ind)))
        f.flush()
        os.fsync(f.fileno())

    print("📁 Melhor indivíduo (média alta + desvio baixo) guardado em 'best_individual.txt'")
    print(f"🏆 Média: {best_mean:.2f} | Desvio padrão: {best_std:.2f}")
    return best_final_ind, best_mean
