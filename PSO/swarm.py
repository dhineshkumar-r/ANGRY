import Utils
import numpy as np
from PSO import Particle


class Swarm:

    def __init__(self, n_features, n_particles, n_iterations, w, c1, c2, sum_size):
        self.n_features = n_features
        self.iter = n_iterations
        self.particles = [Particle(self.n_features) for _ in range(n_particles)]
        self.g_best = Particle(self.n_features)
        self.sum_size = sum_size
        self.run_sum = np.full(self.n_features, 0, dtype=int)
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.features = None
        self.ref_sum = None

    def fitness(self, p):
        p_sum = np.argsort(np.dot(self.features, p.pos))[-self.sum_size:]
        return Utils.calculate_rouge(p_sum, self.ref_sum, 1)

    # TODO Dynamically updating inertia.

    def train(self, features, ref_sum):

        self.features = features
        self.ref_sum = ref_sum

        for _ in range(self.iter):

            # Generate summaries w.r.t particles and find the particle that gives best summary.
            for p in self.particles:
                f_v = self.fitness(p)
                if f_v > p.f_value:
                    p.f_value = f_v
                    p.p_best_pos = p.pos
                if f_v > self.g_best.f_value:
                    self.g_best = p

            # Update other particles' variables with respect to the best particle.
            for p in self.particles:
                p.update_particle(self.g_best, self.w, self.c1, self.c2)

            # Update running sum of features
            self.run_sum += self.g_best.pos

        return self.run_sum / self.iter


if __name__ == "__main__":
    s = Swarm(5, 10, 5, 0.5, 1, 1, 3)
    weights = s.train()
    print(weights)


