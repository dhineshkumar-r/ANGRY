import Utils
import numpy as np
from .particle import Particle
from .constants import *


class Swarm2:

    def __init__(self, documents, ref_sums, n_features=NUMBER_OF_FEATURES, n_particles=NUMBER_OF_PARTICLES,
                 n_iterations=NUMBER_OF_ITERATIONS, w=0.9, c1=C1, c2=C2, sum_size=SUMMARY_SIZE):
        self.documents = documents
        self.ref_sums = ref_sums
        self.n_features = n_features
        self.max_iter = n_iterations
        self.particles = [Particle(self.n_features) for _ in range(n_particles)]
        self.g_best = Particle(self.n_features)
        self.sum_size = sum_size
        self.run_sum = np.full(self.n_features, 0, dtype=int)
        self.w = w
        self.w_min = W_MIN
        self.w_max = W_MAX
        self.c1 = c1
        self.c2 = c2
        self.features = None

    def fitness(self, p):
        rouge_scores = [0.0] * len(self.documents)
        for i, feature in enumerate(self.features):
            p_sum_idx = np.argsort(np.dot(feature, p.pos))[-self.sum_size:]
            p_sum = Utils.join_sentences([self.documents[i][idx] for idx in p_sum_idx])
            rouge_scores[i] = Utils.calculate_rouge(p_sum, self.ref_sums[i], 1)
        return sum(rouge_scores)

    def __update_inertia(self, i):
        return self.w_max - (self.w_max - self.w_min) * (i / self.max_iter)

    def train(self, features):

        self.features = features

        for _ in range(self.max_iter):

            # Generate summaries w.r.t particles and find the particle that gives best summary.
            for p in self.particles:
                f_v = self.fitness(p)
                if f_v > p.f_value:
                    p.f_value = f_v
                    p.p_best_pos = p.pos
                if f_v > self.g_best.f_value:
                    self.g_best = p

            print("Iteration: " + str(_ + 1) + " Fitness: " + str(self.g_best.f_value) + " Best feature: ",
                  self.g_best.pos)

            self.w = self.__update_inertia(_)

            # Update other particles' variables with respect to the best particle.
            for p in self.particles:
                p.update_particle(self.g_best, self.w, self.c1, self.c2)

            # Update running sum of features
            self.run_sum += self.g_best.pos

        return self.run_sum / self.max_iter


class Swarm:

    def __init__(self, document, ref_sum, n_features=NUMBER_OF_FEATURES, n_particles=NUMBER_OF_PARTICLES,
                 n_iterations=NUMBER_OF_ITERATIONS, w=0.9, c1=C1, c2=C2, sum_size=SUMMARY_SIZE):
        self.document = document
        self.ref_sum = Utils.join_docs(ref_sum)
        self.n_features = n_features
        self.max_iter = n_iterations
        self.particles = [Particle(self.n_features) for _ in range(n_particles)]
        self.g_best = Particle(self.n_features)
        self.sum_size = sum_size
        self.run_sum = np.full(self.n_features, 0, dtype=int)
        self.w = w
        self.w_min = W_MIN
        self.w_max = W_MAX
        self.c1 = c1
        self.c2 = c2
        self.features = None

    def fitness(self, p):
        p_sum_idx = np.argsort(np.dot(self.features, p.pos))[-self.sum_size:]
        p_sum = Utils.join_sentences([self.document[idx] for idx in p_sum_idx])
        return Utils.calculate_rouge(p_sum, self.ref_sum, 1)

    def __update_inertia(self, i):
        return self.w_max - (self.w_max - self.w_min) * (i / self.max_iter)

    def train(self, features):

        self.features = features

        for _ in range(self.max_iter):

            # Generate summaries w.r.t particles and find the particle that gives best summary.
            for p in self.particles:
                f_v = self.fitness(p)
                if f_v > p.f_value:
                    p.f_value = f_v
                    p.p_best_pos = p.pos
                if f_v > self.g_best.f_value:
                    self.g_best = p

            print("Iteration: " + str(_ + 1) + " Rouge score: " + str(self.g_best.f_value) + " Best feature: ",
                  self.g_best.pos)

            self.w = self.__update_inertia(_)

            # Update other particles' variables with respect to the best particle.
            for p in self.particles:
                p.update_particle(self.g_best, self.w, self.c1, self.c2)

            # Update running sum of features
            self.run_sum += self.g_best.pos

        return self.run_sum / self.max_iter


if __name__ == "__main__":
    s = Swarm(5, 10, 5, 0.5, 1, 1, 3)
    weights = s.train()
    print(weights)
