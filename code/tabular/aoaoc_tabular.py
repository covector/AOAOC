import gym
import os
import argparse
import numpy as np
from fourrooms import Fourrooms
from scipy.special import logsumexp
from scipy.special import expit

class Tabular:
    def __init__(self, nstates):
        self.nstates = nstates

    def __call__(self, state):
        return np.array([state,])

    def __len__(self):
        return self.nstates


class EgreedyPolicy:
    def __init__(self, rng, nfeatures, noptions, epsilon, weights):
        self.rng = rng
        self.epsilon = epsilon
        self.noptions = noptions
        self.weights = weights

    def value(self, phi, action=None):
        if action is None:
            return np.sum(self.weights[phi, :], axis=0)
        return np.sum(self.weights[phi, action], axis=0)

    def sample(self, phi):
        if self.rng.uniform() < self.epsilon:
            return int(self.rng.randint(self.weights.shape[1]))
        return int(np.argmax(self.value(phi)))


class SoftmaxPolicy:
    def __init__(self, rng, nfeatures, nactions, temp, lr, qWeight):
        self.rng = rng
        self.nactions = nactions
        self.temp = temp
        self.weights = np.zero((nfeatures, noptions))
        self.qWeight = qWeight
        self.lr =lr

    def value(self, phi, action=None):
        if action is None:
            return np.sum(self.weights[phi, :], axis=0)
        return np.sum(self.weights[phi, action], axis=0)

    def pmf(self, phi):
        v = self.value(phi)/self.temp
        return np.exp(v - logsumexp(v))

    def sample(self, phi):
        return int(self.rng.choice(self.nactions, p=self.pmf(phi)))

    def update(self, traject, baseline=None):
        actions_pmf = self.pmf(phi)
        critic = qWeight[traject[0][0], traject[0][1], traject[2]]
        if baseline:
            critic -= baseline
        self.weights[traject[0][0], :] -= self.lr*critic*actions_pmf
        self.weights[traject[0][0], traject[2]] += self.lr*critic


class Q_U:
    def __init__(self, nfeatures, nactions, discount, lr, weights, policy_over_options):
        self.weights = weights
        self.lr = lr
        self.discount = discount
        self.policy_over_options = policy_over_options
    
    def value(self, phi, option, action):
        return np.sum(self.weights[phi, option, action], axis=0)

    def update(self, traject, reward, done, termination):
        # One-step update target
        update_target = reward
        if not done:
            current_values = self.policy_over_options.value(traject[1][0])
            update_target += self.discount*((1. - termination)*current_values[traject[0][1]] + termination*np.max(current_values))

        # Update values upon arrival if desired
        tderror = update_target - self.value(traject[0][0], traject[0][1], traject[2])
        self.weights[traject[0][0], traject[0][1], traject[2]] += self.lr*tderror

class Q_O:
    def __init__(self, nfeatures, noptions, discount, lr, weights):
        self.weights = weights
        self.lr = lr
        self.discount = discount

    def value(self, phi, option=None):
        if option is None:
            return np.sum(self.weights[phi, :], axis=0)
        return np.sum(self.weights[phi, option], axis=0)

    def update(self, traject, reward, done, termination):
        # One-step update target
        update_target = reward
        if not done:
            current_values = self.value(phi)
            update_target += self.discount*((1. - termination)*current_values[traject[0][1]] + termination*np.max(current_values))

        # Dense gradient update step
        tderror = update_target - self.last_value
        self.weights[traject[0][0],traject[0][1]]] += self.lr*tderror


class SigmoidTermination:
    def __init__(self, rng, nfeatures, lr, policy_over_options):
        self.rng = rng
        self.weights = np.zeros((nfeatures,))
        self.lr = lr
        self.policy_over_options = policy_over_options

    def pmf(self, phi):
        return expit(np.sum(self.weights[phi]))

    def sample(self, phi):
        return int(self.rng.uniform() < self.pmf(phi))

    def grad(self, phi):
        terminate = self.pmf(phi)
        return terminate*(1. - terminate), phi
    
    def update(self, phi, option):
        magnitude, direction = self.grad(phi)
        self.weights[direction] -= self.lr*magnitude*(self.policy_over_options.advantage(phi, option))

class SigmoidAttention:
    def __init__(self, nactions, qWeight):
        self.weights = np.zeros((nactions,))
        self.qWeight = qWeight

    def pmf(self):
        return expit(np.sum(self.weights), axis=0)

    def grad(self, phi):
        attend = self.pmf()
        return attend*(1. - attend)

    def attention(self, a):
        return self.pmf[a]

    def update(self, traject):
        pass

class PredefinedAttention:
    def __init__(self, index):
        if (index==0):
            self.weights = np.array([1,1,0,0])
        if (index==1):
            self.weights = np.array([0,1,1,0])
        if (index==2):
            self.weights = np.array([0,0,1,1])
        if (index==3):
            self.weights = np.array([1,0,0,1])

    def pmf(self):
        return np.sum(self.weights, axis=0)

    def attention(self, a):
        return self.pmf[a]

    def update(self, traject):
        pass

class FinalPolicy:
    def __init__(self, rng, nfeatures, nactions, temp, lr_a, qWeight, lr_p, qWeightp):
        self.rng = rng
        self.nfeatures = nfeatures
        self.nactions = nactions
        self.EorS = EorS
        self.internalPI = SoftmaxPolicy(rng, nfeatures, nactions, temp, lr_p, qWeight)
        # self.attention = SigmoidAttention(nactions, lr_a, qWeight)
        self.attention = PredefinedAttention()

    def pmf(self, phi):
        pi = self.internalPI.pmf(phi)
        h = self.attention.pmf()
        normalizer = np.dot(pi, h)
        return np.multiply(pi,h)/normalizer

    def sample(self, phi):
        return int(self.rng.choice(self.nactions, p=self.pmf(phi)))

    def H_update(self, traject):
        self.attention.update(traject)

    def P_update(self, traject, baseline=None):
        self.internalPI.update(traject, baseline)

class Distraction:
    def __init__(self, xi, R, n, h):
        self.xi = xi
        self.R = R
        self.n = n
        self.h = h

    def reward(self, r, a):
        return r - self.xi*self.R(1-(self.h.attention(a))**self.n)


class Option:
    def __init__(self, rng, nfeatures, nactions, args, R, policy_over_options):
        self.weights = np.zero((nfeatures, nactions)) 
        self.weightsP = np.zero((nfeatures, nactions)) 
        self.policy = FinalPolicy(rng, nfeatures, nactions, args.temperature, args.lr_attend, weights, lr_intra, qWeightp)
        self.termination = SigmoidTermination(rng, nfeatures, args.lr_term, policy_over_options)
        self.Qval = Q_U(nfeatures, nactions, args.discount, args.lr_critic, weights, policy_over_options)
        self.PQval = Q_U(nfeatures, nactions, args.discount, args.lr_critic_pseudo, weightsP, policy_over_options)
        self.distraction = Distraction(args.xi, R, args.n, self.policy.attention)
        self.baseline = args.baseline

    def sample(self, phi):
        return self.policy.sample(phi)

    def distract(self, reward, action):
        return self.distraction.reward(reward, action)

    def terminate(self, phi):
        return self.termination.sample(phi)

    def Q_update(self, traject, reward, done):
        self.Q_U.update(traject, reward, done)
        self.P_Q_U.update(traject, self.distraction(reward), done)

    def H_update(self, traject):
        self.policy.H_update(traject)

    def B_update(self, phi, option):
        self.termination.update(phi, option)

    def P_update(self, traject)
        if self.baseline:
            self.policy.P_update(traject, policy_over_options.value(traject[0][0], traject[0][1]))
        else:
            self.policy.P_update(traject)

    def update(self, traject, reward, done, phi, option):
        Q_update(traject, reward, done)
        H_update(traject)
        P_update(traject)
        B_update(phi, option)


class POO:
    def __init__(self, rng, nfeatures, args, R)
        self.weights = np.zero((nfeatures, args.noptions))
        self.weightsP = np.zero((nfeatures, args.noptions))
        self.policy = EgreedyPolicy(rng, nfeatures, args.noptions, args.epsilon, self.weights)
        self.Q_Omega = Q_O(nfeatures, args.noptions, args.discount, args.lr_critic, self.weights)
        self.Q_OmegaP = Q_O(nfeatures, args.noptions, args.discount, args.lr_critic_pseudo, self.weightsP)

    def update(self, traject, reward, distracted, done, termination):
        self.Q_Omega.update(traject, reward, done, termination)
        self.Q_OmegaP.update(traject, distracted, done, termination)

    def sample(self, phi):
        return self.policy.sample(phi)

    def advantage(self, phi, option=None):
        values = self.weightsP(phi)
        advantages = values - np.max(values)
        if option is None:
            return advantages
        return advantages[option]

    def value(self, phi, option=None):
        if option is None:
            return np.sum(self.weights[phi, :], axis=0)
        return np.sum(self.weights[phi, option], axis=0)


def save_params(args, dir_name):
    f = os.path.join(dir_name, "Params.txt")
    with open(f, "w") as f_w:
        for param, val in sorted(vars(args).items()):
            f_w.write("{0}:{1}\n".format(param, val))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--discount', help='Discount factor', type=float, default=0.99)
    parser.add_argument('--lr_term', help="Termination gradient learning rate", type=float, default=0.25)
    parser.add_argument('--lr_intra', help="Intra-option gradient learning rate", type=float, default=0.25)
    parser.add_argument('--lr_critic', help="Learning rate of Q_Omega", type=float, default=0.5)
    parser.add_argument('--lr_critic_pseudo', help="Learning rate of pseudo Q_Omega", type=float, default=0.5)
    parser.add_argument('--lr_criticA', help="Learning rate of Q_U", type=float, default=0.5)
    parser.add_argument('--lr_criticA_pseudo', help="Learning rate of pseudo Q_U", type=float, default=0.5)
    parser.add_argument('--lr_attend', help="Leraning rate of attention mech", type=float, deafult=0.5)
    parser.add_arguement('--h_learn', help="Learnable attention mech", action='store_true')
    parser.add_argument('--xi', help="Exploratory", type=float, default=1.)
    parser.add_argument('--n', help="Power", type=float, default=0.5)
    parser.add_argument('--epsilon', help="Epsilon-greedy", type=float, default=1e-2)
    parser.add_argument('--nepisodes', help="Number of episodes per run", type=int, default=2000)
    parser.add_argument('--nruns', help="Number of runs", type=int, default=1)
    parser.add_argument('--nsteps', help="Maximum number of steps per episode", type=int, default=2000)
    parser.add_argument('--noptions', help='Number of options', type=int, default=4)
    parser.add_argument('--baseline', help="Use the baseline for the intra-option gradient", action='store_true')
    parser.add_argument('--temperature', help="Temperature parameter for softmax", type=float, default=1e-2)
    parser.add_argument('--seed', help="seed value for experiment", type=int, default=10)
    parser.add_argument('--seed_startstate', help="seed value for starting state", type=int, default=10)

    args = parser.parse_args()
    rng = np.random.RandomState(args.seed)
    env = Fourrooms(args.seed_startstate)
    R = 50.

    outer_dir = "AOAOC"
    if not os.path.exists(outer_dir):
        os.makedirs(outer_dir)

    dir_name = "Runs"+str(args.nruns)+"_Epsds"+str(args.nepisodes)+ "_Eps"+str(args.epsilon)+"_NOpt"+str(args.noptions)+"_LRT"+ str(args.lr_term) +\
     "_LRI"+str(args.lr_intra)+"_LRC"+str(args.lr_critic)+"_LRCP"+str(args.lr_critic_pseudo)+"_Xi"+str(args.xi)+"_N"+str(args.n)+\
     "_temp"+str(args.temperature)+"_seed"+str(args.seed)

    dir_name = os.path.join(outer_dir, dir_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    save_params(args, dir_name)

    possible_next_goals = [68, 69, 70, 71, 72, 78, 79, 80, 81, 82, 88, 89, 90, 91, 92, 93, 99, 100, 101,102, 103]

    num_states = env.observation_space.n
    num_actions = env.action_space.n

    history = np.zeros((args.nruns, args.nepisodes, 3), dtype=np.float32)

    state_frequency_history = np.zeros((args.nruns, args.nepisodes, env.observation_space.n, args.noptions),dtype=np.int32)


    for run in range(args.nruns):
        features = Tabular(env.observation_space.n)
        nfeatures, nactions = len(features), env.action_space.n

        policy_over_options = POO(rng, nfeatures, args, R)

        options = [Option(rng, nfeatures, nactions, args, R, policy_over_options) for _ in range(args.noptions)]


        for episode in range(args.nepisodes):
            return_per_episode = 0.0
            observation = env.reset()

            phi = features(observation)

            option = policy_over_options.sample(phi)

            action = options[option].sample(phi)

            traject = [[phi,option],[phi,option],action]

            cumreward = 0.
            duration = 1
            option_switches = 0
            avgduration = 0.
            for step in range(args.nsteps):
                observation, reward, done, _ = env.step(action)
                phi = features(observation)
                return_per_episode += pow(args.discount,step)*reward

                state_frequency_history[run, episode, observation, option] +=1

                last_option = option

                if option[option].terminate(phi):
                    option = policy_over_options.sample(phi)
                    option_switches += 1
                    avgduration += (1./option_switches)*(duration - avgduration)
                    duration = 1

                termination = options[option].termination.pmf(phi)

                traject[0] = traject[1]
                traject[1] = [phi, option]
                traject[2] = action

                action = options[option].sample(phi)

                options[last_option].update(traject, reward, done, traject[1][0], last_option)

                policy_over_options.update(traject, reward, options[last_option].distract(reward,traject[2]), done, termination)

                cumreward += reward
                duration += 1
                if done:
                    break

            history[run, episode, 0] = step
            history[run, episode, 1] = return_per_episode
            history[run, episode, 2] = avgduration

            print('Run {} episode {} steps {} cumreward {} avg. duration {} switches {}'.format(run, episode, step, cumreward, avgduration, option_switches))