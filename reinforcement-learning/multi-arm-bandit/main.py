import random
import numpy as np
import enum


class MultiArmBandit:

    def __init__(self, q: [float]):
        self.q = q

    def get_reward(self, action_index: int):
        return self.q[action_index]

    def update_q_star(self, mu=0, sigma=0.01):
        for idx, value in enumerate(self.q):
            value += np.random.normal(mu, sigma, 1)[0]
            self.q[idx] = value


class Type(enum.Enum):
    SAMPLE_AVERAGE = 1
    CONSTANT_STEP_SIZE = 2


class MultiArmBanditLearner:

    def __init__(self, q_estimate: [float], epsilon: float, t: Type, constant_step_size=None):
        self.q_estimate = q_estimate
        self.epsilon = epsilon
        self.reward_total = 0.0

        self.type = t
        if t == Type.SAMPLE_AVERAGE:
            self.n = [0] * len(self.q_estimate)
        elif t == Type.CONSTANT_STEP_SIZE:
            self.constant_step_size = constant_step_size

    def get_step_size(self, index: int):
        if self.type == Type.SAMPLE_AVERAGE:
            self.n[index] += 1
            n = self.n[index]
            return 1/n
        elif self.type == Type.CONSTANT_STEP_SIZE:
            return self.constant_step_size

    def step(self, env: MultiArmBandit):
        if random.uniform(0, 1) < self.epsilon:
            index = random.randint(0, len(self.q_estimate) - 1)
        else:
            index = self.q_estimate.index(max(self.q_estimate))

        r = env.get_reward(index)
        q = self.q_estimate[index]
        s = self.get_step_size(index)
        self.q_estimate[index] = q + s * (r - q)

        self.reward_total += r


class EnvironmentType(enum.Enum):
    STATIC = 1
    DYNAMIC = 2


if __name__ == "__main__":

    def run(t: EnvironmentType):
        q_real = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        environment = MultiArmBandit(q_real)

        eps = 0.1
        ss = 0.1
        initial_q_estimate = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        rl_css = MultiArmBanditLearner(initial_q_estimate.copy(), eps, Type.CONSTANT_STEP_SIZE, ss)
        rl_sa  = MultiArmBanditLearner(initial_q_estimate.copy(), eps, Type.SAMPLE_AVERAGE, None)

        for i in range(10000):
            rl_css.step(environment)
            rl_sa.step(environment)

            if t == EnvironmentType.DYNAMIC:
                environment.update_q_star()

        print("rl constant step size - reward total: " + str(rl_css.reward_total))
        print("rl sample average     - reward total: " + str(rl_sa.reward_total))

    # Static Environment
    print("Environment Values Static - RL Sample Average should have higher reward total in theory")
    run(EnvironmentType.STATIC)

    # Dynamic Environment
    print("\nEnvironment Values Dynamic - RL Constant Step Size should have higher reward total in theory")
    run(EnvironmentType.DYNAMIC)
