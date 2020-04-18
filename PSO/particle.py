import numpy as np
import Utils
from .constants import *


class Particle:

    def __init__(self, n_features):
        self.pos = np.array(Utils.randBinList(n_features))
        self.vel = np.full(n_features, 0)
        self.v_max = V_MAX
        self.v_min = V_MIN
        self.p_best_pos = self.pos
        self.n_feat = n_features
        self.f_value = float("-inf")

    def __update_position(self):
        u_vel = Utils.sigmoid(self.vel)
        r = Utils.get_random_nos(self.n_feat)
        self.pos = ((r - u_vel) < 0) * 1
        return

    def __update_velocity(self, g_best, w, c1, c2):
        r1 = Utils.get_random_nos(self.n_feat)
        r2 = Utils.get_random_nos(self.n_feat)
        self.vel = w * self.vel + c1 * r1 * (self.p_best_pos - self.pos) + c2 * r2 * (g_best.pos - self.pos)
        self.vel = np.clip(self.vel, a_min=self.v_min, a_max=self.v_max)
        return

    def update_particle(self, g_best, w, c1, c2):
        self.__update_velocity(g_best, w, c1, c2)
        self.__update_position()
        return

    def print_particle(self):
        print("Position: ", self.pos)
        print("Velocity: ", self.vel)
        print("Best position:", self.p_best_pos)
        return


if __name__ == "__main__":
    p = Particle(3)
    p2 = Particle(3)
    p2.pos = [1, 0, 1]
    p.update_particle(p2, 0.5, 1, 1)
    p.print_particle()
    p2.print_particle()
