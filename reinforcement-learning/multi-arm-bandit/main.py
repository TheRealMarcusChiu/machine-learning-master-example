import random
import numpy as np
import enum


class MultiArmBandit:

    def __init__(self, q: [float]):
        self.q = q

    def execute_action(self, action_index: int):
        # return self.q[action_index]
        return np.random.normal(self.q[action_index], 1, 1)[0]

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
        elif t == Type.CONSTANT_STEP_SIZE:
            self.q_estimate = q_estimate
            self.constant_step_size = constant_step_size

        self.ucb = False
        self.ucb_c_value = 0
        self.total_steps = 0
        self.n = [0] * len(self.q_estimate)

    # using ucb would ignore epsilon
    def with_ucb(self, ucb_c_value=2):
        self.ucb = True
        self.ucb_c_value = ucb_c_value
        self.total_steps = 0
        return self

    def get_ucb_index(self):
        if self.n[0] == 0:
            return 0

        maximum = self.q_estimate[0] + (self.ucb_c_value * np.sqrt(np.log(self.total_steps)/self.n[0]))
        index = 0

        for idx, q in enumerate(self.q_estimate):
            if self.n[idx] == 0:
                index = idx
                break
            else:
                m = q + (self.ucb_c_value * np.sqrt(np.log(self.total_steps)/self.n[idx]))
                if m > maximum:
                    maximum = m
                    index = idx

        return index

    def choose_action_index(self):
        if self.ucb:
            self.total_steps += 1
            index = self.get_ucb_index()
        else:
            if random.uniform(0, 1) < self.epsilon:
                index = random.randint(0, len(self.q_estimate) - 1)
            else:
                index = self.q_estimate.index(max(self.q_estimate))
        return index

    def get_step_size(self, index: int):
        if self.type == Type.SAMPLE_AVERAGE:
            return 1/self.n[index]
        elif self.type == Type.CONSTANT_STEP_SIZE:
            return self.constant_step_size

    def step(self, env: MultiArmBandit):
        action_index = self.choose_action_index()
        self.n[action_index] += 1

        r = env.execute_action(action_index)
        q = self.q_estimate[action_index]
        s = self.get_step_size(action_index)
        self.q_estimate[action_index] = q + s * (r - q)

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
        # rl_sa
        # sample average

        # rl_css_5
        # Optimistic Initial Values
        # initial q estimate values = 5, therefore encourages exploration temporarily in beginning,
        # bc true q values are mean=0 and var=std=1
        # The result is that all actions are tried several times before the value estimates converge.
        # The system does a fair amount of exploration even if greedy actions are selected all the time
        # look at (Constant Step Size - Initial Q Estimates Graph.png)

        rl_sa          = MultiArmBanditLearner([0] * size, eps, Type.SAMPLE_AVERAGE, None)
        rl_sa_ucb_2    = MultiArmBanditLearner([0] * size, eps, Type.SAMPLE_AVERAGE, None).with_ucb(2)
        rl_css_0       = MultiArmBanditLearner([0] * size, eps, Type.CONSTANT_STEP_SIZE, ss)
        rl_css_5       = MultiArmBanditLearner([5] * size, eps, Type.CONSTANT_STEP_SIZE, ss)
        rl_css_5_ucb_2 = MultiArmBanditLearner([5] * size, eps, Type.CONSTANT_STEP_SIZE, ss).with_ucb(2)

        for i in range(100000):
            rl_sa.step(environment)
            rl_sa_ucb_2.step(environment)
            rl_css_0.step(environment)
            rl_css_5.step(environment)
            rl_css_5_ucb_2.step(environment)

            if t == EnvironmentType.DYNAMIC:
                environment.update_q_star()

        print("- sample average                                              reward total: " + str(rl_sa.reward_total))
        print("- sample average with UCB (c=2)                               reward total: " + str(rl_sa_ucb_2.reward_total))
        print("- constant step size (initial q estimates = 0)                reward total: " + str(rl_css_0.reward_total))
        print("- constant step size (initial q estimates = 5)                reward total: " + str(rl_css_5.reward_total))
        print("- constant step size (initial q estimates = 5) with UCB (c=2) reward total: " + str(rl_css_5_ucb_2.reward_total))


    # Static Environment
    print("Multi-Armed Bandit - STATIC")
    run(EnvironmentType.STATIC)

    # Dynamic Environment
    print("\nMulti-Armed Bandit - DYNAMIC")
    run(EnvironmentType.DYNAMIC)
