import Utils
import numpy as np
from .particle import Particle
from .constants import *
from typing import List


class Swarm:

    def __init__(self, documents: List[List[List[str]]], ref_sums: List[str], n_features=NUMBER_OF_FEATURES, n_particles=NUMBER_OF_PARTICLES,
                 n_iterations=NUMBER_OF_ITERATIONS, w=0.9, c1=C1, c2=C2, sum_size=SUMMARY_SIZE, config=None):
        self.config = config
        self.documents: List[List[List[str]]] = documents
        self.ref_sums: List[str] = ref_sums
        self.n_features = n_features
        self.max_iter = n_iterations
        self.particles: List[Particle] = [Particle(self.n_features) for _ in range(n_particles)]
        self.g_best = Particle(self.n_features)
        self.sum_size = sum_size
        self.run_sum = np.full(self.n_features, 0, dtype=int)
        self.w = w
        self.w_min = config.w_min
        self.w_max = config.w_min
        self.c1 = c1
        self.c2 = c2
        self.features = None

    def fitness(self, p: Particle) -> float:
        rouge_scores = [0.0] * len(self.documents)
        for i, feature in enumerate(self.features):
            p_sum_idx = np.argsort(np.dot(feature, p.pos))[-self.sum_size:]
            p_sum: str = Utils.join_sentences([self.documents[i][idx] for idx in p_sum_idx])
            rouge_scores[i] = Utils.calculate_rouge(p_sum, [self.ref_sums[i]], 1)
        return sum(rouge_scores)/len(rouge_scores)

    def __update_inertia(self, i) -> float:
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

            print("Iteration: " + str(_ + 1) + " Fitness: " + str(self.g_best.f_value) + " Best feature combination: ",
                  self.g_best.pos)

            self.w = self.__update_inertia(_)

            # Update other particles' variables with respect to the best particle.
            for p in self.particles:
                p.update_particle(self.g_best, self.w, self.c1, self.c2)

            # Update running sum of features
            self.run_sum += self.g_best.pos

        return self.run_sum / self.max_iter
