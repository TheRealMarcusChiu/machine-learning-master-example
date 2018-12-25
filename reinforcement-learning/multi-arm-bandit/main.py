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
        self.epsilon = epsilon
        self.reward_total = 0.0

        self.type = t
        if t == Type.SAMPLE_AVERAGE:
            self.q_estimate = [0] * len(q_estimate)
            self.n = [0] * len(self.q_estimate)
        elif t == Type.CONSTANT_STEP_SIZE:
            self.q_estimate = q_estimate
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

    mean = 0
    std = 1
    size = 10
    environment = MultiArmBandit(np.random.normal(mean, std, size))
    print(environment.q)

    eps = 0.1
    ss = 0.1


    def run(t: EnvironmentType):

        rl_css_0 = MultiArmBanditLearner([0] * size, eps, Type.CONSTANT_STEP_SIZE, ss)

        # initial q estimate values = 5, therefore encourages exploration, bc true q values are mean=0 and var=std=1
        # The result is that all actions are tried several times before the value estimates converge.
        # The system does a fair amount of exploration even if greedy actions are selected all the time
        # look at (Constant Step Size - Initial Q Estimates Graph.png)
        rl_css_5 = MultiArmBanditLearner([5] * size, eps, Type.CONSTANT_STEP_SIZE, ss)
        rl_sa  = MultiArmBanditLearner([0] * size, eps, Type.SAMPLE_AVERAGE, None)

        for i in range(10000):
            rl_css_0.step(environment)
            rl_css_5.step(environment)
            rl_sa.step(environment)

            if t == EnvironmentType.DYNAMIC:
                environment.update_q_star()

        print("constant step size (initial q estimates = 0) reward total: " + str(rl_css_0.reward_total))
        print("constant step size (initial q estimates = 5) reward total: " + str(rl_css_5.reward_total))
        print("sample average                               reward total: " + str(rl_sa.reward_total))


    # Static Environment
    print("Static Multi-Armed Bandit - Sample Average Learner in theory should have higher reward total")
    run(EnvironmentType.STATIC)

    # Dynamic Environment
    print("\nDynamic Multi-Armed Bandit - Constant Step Size Learner in theory should have higher reward total")
    run(EnvironmentType.DYNAMIC)
