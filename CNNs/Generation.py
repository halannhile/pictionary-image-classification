from Model import Model
import random

class Generation:
    def __init__(self, population_size, survival_rate, gen_no):
        self.population_size = population_size
        self.survival_rate = survival_rate
        self.population = []
        self.gen_no = gen_no

    """
    Will randomly generate models 
    """
    def generate_population(self):
        self.population = [Model([784,6]) for _ in range(self.population_size)]
        for model in self.population:
            model.randomise_properties()
    """
    Sets the population (needed when evolving)
    """
    def set_population(self,population):
        self.population = population

    def __str__(self):
        output =""
        for model in self.population:
            output += str(model) + "\n"

        output += "\n"

        return  output

    """
    Evolves the generation and returns a new generation
    """
    def evolve(self, data):
        results = []

        # for each model calculate score
        for model in self.population:
            results.append([model.score(data), model])

        # sort scores
        sorted_results = sorted(results, key=lambda x: x[0], reverse=True)

        next_population = []

        # calculate number of survivors and add top n models to next population
        survivors = int(self.survival_rate * self.population_size)
        for i in range(survivors):
            next_population.append(sorted_results[i][1])

        # calculate number of new models required and reproduce survivors
        children_required = self.population_size - survivors
        for i in range(children_required):
            r1, r2 = self.chose_two_parents(survivors)
            child = next_population[r1].reproduce(next_population[r2])
            next_population.append(child)

        # saves outcome to a txt file (for debugging)
        self.save_generation_results(sorted_results, next_population)

        # create new generation and set population
        next_generation = Generation(self.population_size,self.survival_rate, self.gen_no +1)
        next_generation.set_population(next_population)

        return next_generation

    def chose_two_parents(self, no_parents):
        r1 = 0
        r2 = 0
        while r1 == r2:
            r1 = random.randint(0, no_parents - 1)
            r2 = random.randint(0, no_parents - 1)

        return (r1, r2)

    def save_generation_results(self, sorted_population, next_population):
        outupt = ["Results for generation {}\n".format(self.gen_no)]
        outupt.append("older generation results:")
        for i in range(len(sorted_population)):
            model = sorted_population[i][1]
            outupt.append("({}) - accuracy: {}".format(i, model.score({})))
            outupt.append(str(model))

        outupt.append("new generation:")
        for i in range(len(next_population)):
            model = next_population[i]
            outupt.append(str(model) )

        with open("gen_{}.txt".format(self.gen_no), 'w') as out_file:
            out_file.writelines(outupt)
            out_file.close()

        print(outupt)




