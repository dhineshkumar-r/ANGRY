import Utils
import numpy as np
from PSO import Particle


def join_sentences(doc):
    return " ".join([" ".join(_) for _ in doc])


def join_docs(docs):
    return [join_sentences(d) for d in docs]


class Swarm:

    def __init__(self, document, ref_sum, n_features=3, n_particles=3, n_iterations=100, w=0.5, c1=1, c2=1,
                 sum_size=75):
        self.document = document
        self.ref_sum = join_docs(ref_sum)
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
        p_sum_idx = np.argsort(np.dot(self.features, p.pos))[-self.sum_size:]
        p_sum = join_sentences([self.document[idx] for idx in p_sum_idx])
        return Utils.calculate_rouge(p_sum, self.ref_sum, 1)

    # TODO Dynamically updating inertia.

    def train(self, features):

        self.features = features

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
