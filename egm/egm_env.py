import numpy as np
import random
import spice_runner


class EgmEnv:
    def __init__(self, waveform=None):
        self.action_space = ['inc_rstrap', 'inc_cstrap', 'inc_lstrap', 'inc_rdelay',
                             'inc_cdelay', 'inc_cbody', 'dec_rstrap', 'dec_cstrap',
                             'dec_lstrap', 'dec_rdelay', 'dec_cdelay', 'dec_cbody', 'nothing']
        self.n_actions = len(self.action_space)
        self.observation_space = 6 + 300 * 2
        self.rstrap = random.uniform(50, 550)
        self.cstrap = random.uniform(1e-12, 100e-12)
        self.lstrap = random.uniform(0.1e-6, 10e-6)
        self.rdelay = random.uniform(50, 550)
        self.cdelay = random.uniform(1e-12, 60e-12)
        self.cbody = random.uniform(1e-12, 600e-12)
        self.target_waveform = None
        self.current_waveform = None
        self.step_count = 0

    def reset(self, target_waveform=None):
        self.step_count = 0
        self.rstrap = random.uniform(50, 550)
        self.cstrap = random.uniform(1e-12, 100e-12)
        self.lstrap = random.uniform(0.1e-6, 10e-6)
        self.rdelay = random.uniform(50, 550)
        self.cdelay = random.uniform(1e-12, 60e-12)
        self.cbody = random.uniform(1e-12, 600e-12)
        self.current_waveform = self.get_current_waveform()

        if target_waveform is None:
            _rstrap = random.uniform(50, 550)
            _cstrap = random.uniform(1e-12, 100e-12)
            _lstrap = random.uniform(0.1e-6, 10e-6)
            _rdelay = random.uniform(50, 550)
            _cdelay = random.uniform(1e-12, 60e-12)
            _cbody = random.uniform(1e-12, 600e-12)
            self.target_waveform = spice_runner.get_waveform(_rstrap, _cstrap, _lstrap,
                                                             _rdelay, _cdelay, _cbody)
        else:
            self.target_waveform = target_waveform
        return self.get_state()

    def get_current_waveform(self):
        waveform = spice_runner.get_waveform(self.rstrap, self.cstrap, self.lstrap,
                                             self.rdelay, self.cdelay, self.cbody)
        return waveform

    def get_state(self):
        state = [self.rstrap, self.cstrap, self.lstrap, self.rdelay, self.cdelay, self.cbody]
        state += list(self.current_waveform)
        state += list(self.target_waveform)
        return state

    @staticmethod
    def calculate_similarity(waveform1, waveform2):
        if waveform1.shape != waveform2.shape:
            print("Different shape")
            return 0.0
        similarity = -np.sum(np.square(waveform1 - waveform2))
        return similarity

    def step(self, action):
        self.step_count += 1
        if action == 0:
            self.rstrap *= 2
        elif action == 1:
            self.cstrap *= 2
        elif action == 2:
            self.lstrap *= 2
        elif action == 3:
            self.rdelay *= 2
        elif action == 4:
            self.cdelay *= 2
        elif action == 5:
            self.cbody *= 2
        elif action == 6:
            self.rstrap /= 2
        elif action == 7:
            self.cstrap /= 2
        elif action == 8:
            self.lstrap /= 2
        elif action == 9:
            self.rdelay /= 2
        elif action == 10:
            self.cdelay /= 2
        elif action == 11:
            self.cbody /= 2
        else:
            pass

        self.current_waveform = self.get_current_waveform()

        reward, done = 0, False
        if action >= 12 or self.step_count >= 100:
            reward = self.calculate_similarity(self.current_waveform, self.target_waveform)
            done = True
        else:
            reward = 0
            done = False

        next_state = [self.rstrap, self.cstrap, self.lstrap, self.rdelay, self.cdelay, self.cbody]
        next_state += list(self.current_waveform)
        next_state += list(self.target_waveform)
        return next_state, reward, done, None

    def render(self):
        import matplotlib.pyplot as plt
        plt.plot(self.current_waveform)
        plt.plot(self.target_waveform)
        plt.show()
        plt.close()


if __name__ == '__main__':
    env = EgmEnv()
    print(env.reset())
    while True:
        env.render()
        action = int(input('ACTION : '))
        state, reward, done = env.step(action)
        print("===State===\n", state)
        print("Reward :", reward, " Done :", done)
