import gym
import os
import argparse
import numpy as np
from fourrooms import Fourrooms
from scipy.special import logsumexp
from scipy.special import expit
from visualize import Display

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

    def _value(self, phi, action=None):
        if action is None:
            return np.sum(self.weights[phi, :], axis=0)
        return np.sum(self.weights[phi, action], axis=0)

    def sample(self, phi):
        if self.rng.uniform() < self.epsilon:
            return int(self.rng.randint(self.weights.shape[1]))
        return int(np.argmax(self._value(phi)))


class SoftmaxPolicy:
    def __init__(self, rng, nfeatures, nactions, temp, lr, qWeight):
        self.rng = rng
        self.nactions = nactions
        self.temp = temp
        self.weights = np.zeros((nfeatures, nactions))
        self.qWeight = qWeight
        self.lr =lr

    def _value(self, phi, action=None):
        if action is None:
            return np.sum(self.weights[phi, :], axis=0)
        return np.sum(self.weights[phi, action], axis=0)

    def pmf(self, phi):
        v = self._value(phi)/self.temp
        return np.exp(v - logsumexp(v))

    def sample(self, phi):
        return int(self.rng.choice(self.nactions, p=self.pmf(phi)))

    def update(self, traject, baseline=None):
        actions_pmf = self.pmf(traject[0][0])
        critic = self.qWeight[traject[0][0], traject[2]]
        if baseline:
            critic -= baseline
        self.weights[traject[0][0], :] -= self.lr*critic*actions_pmf
        self.weights[traject[0][0], traject[2]] += self.lr*critic


class Q_U:
    def __init__(self, nfeatures, nactions, discount, lr, weights, policy_over_options, pseudo):
        self.weights = weights
        self.lr = lr
        self.discount = discount
        self.policy_over_options = policy_over_options
        self.pseudo = pseudo
    
    def value(self, phi, action):
        return np.sum(self.weights[phi, action], axis=0)

    def update(self, traject, reward, done, termination):
        # One-step update target
        update_target = reward
        if not done:
            current_values = self.policy_over_options.value(traject[1][0], pseudo=self.pseudo)
            update_target += self.discount*((1. - termination)*current_values[traject[0][1]] + termination*np.max(current_values))

        # Update values upon arrival if desired
        tderror = update_target - self.value(traject[0][0], traject[2])
        self.weights[traject[0][0], traject[2]] += self.lr*tderror

class Q_O:
    def __init__(self, nfeatures, noptions, discount, lr, weights):
        self.weights = weights
        self.lr = lr
        self.discount = discount

    def _value(self, phi, option=None):
        if option is None:
            return np.sum(self.weights[phi, :], axis=0)
        return np.sum(self.weights[phi, option], axis=0)

    def update(self, traject, reward, done, termination):
        # One-step update target
        update_target = reward
        if not done:
            current_values = self._value(traject[1][0])
            update_target += self.discount*((1. - termination)*current_values[traject[0][1]] + termination*np.max(current_values))

        # Dense gradient update step
        tderror = update_target - self._value(traject[0][0], traject[0][1])
        self.weights[traject[0][0],traject[0][1]] += self.lr*tderror


class SigmoidTermination:
    def __init__(self, rng, nfeatures, lr, policy_over_options, dc):
        self.rng = rng
        self.weights = np.zeros((nfeatures,))
        self.lr = lr
        self.policy_over_options = policy_over_options
        self.dc = dc

    def pmf(self, phi):
        return expit(np.sum(self.weights[phi]))

    def sample(self, phi):
        return int(self.rng.uniform() < self.pmf(phi))

    def _grad(self, phi):
        terminate = self.pmf(phi)
        return terminate*(1. - terminate), phi
    
    def update(self, phi, option):
        magnitude, direction = self._grad(phi)
        self.weights[direction] -= self.lr*magnitude*(self.policy_over_options.advantage(phi, option)+self.dc)


class SigmoidAttention:
    def __init__(self, nactions, lr, qWeight, clipthres, stretchthres, stretchstep):
        self.weights = np.random.uniform(low=-1, high=1, size=(nactions,))
        self.qWeight = qWeight
        self.lr = lr
        self.clipthres = clipthres
        self.stretchthres = stretchthres
        self.stretchstep = stretchstep

    def pmf(self):
        return expit(self.weights)

    def _grad(self):
        attend = self.pmf()
        return attend*(1. - attend)

    def attention(self, a):
        return self.pmf()[a]

    def update(self, traject, gradList):
        # self.weights += self.lr * np.sum(gradList, axis=0) * self._grad()
        self.weights += self.lr * np.sum(np.clip(gradList,-5,5), axis=0) * self._grad()
        for i in range(len(self.weights)):
            if self.weights[i]<-4.5:
                self.weights[i] = -4.5

    def normalize(self):
        normalizer = np.linalg.norm(self.pmf())
        self.weights /= normalizer

    # def regulate(self, o):
    #     normalizer = np.linalg.norm(self.pmf())
    #     if normalizer<self.stretchthres:
    #         o.weight *= self.stretchstep
    #     else:
    #         o.weight /= self.stretchstep
    #     expo = np.exp(-1 * self.weights)
        # self.weights += np.log(expo / (normalizer*(1+expo) - 1))
        # print(np.linalg.norm(self.pmf()))
        # if normalizer<self.clipthres:
        #     print('clip')
        #     self.weights += np.log(self.clipthres/normalizer)
        #     print(np.linalg.norm(self.pmf()))
            
            
        

class PredefinedAttention:
    def __init__(self, index):
        # if (index==0):
        #     self.weights = np.array([1, 0.01, 0.01, 1])
        # if (index==1):
        #     self.weights = np.array([0.01, 1, 0.01, 1])
        # if (index==2):
        #     self.weights = np.array([0.01, 1, 1, 0.01])
        # if (index==3):
        #     self.weights = np.array([1, 0.01, 1, 0.01])
        if (index==0):
            self.weights = np.array([0.01, 1, 0.01, 1])
        if (index==1):
            self.weights = np.array([1, 0.01, 0.01, 1])
        if (index==2):
            self.weights = np.array([0.01, 0.01, 1, 0.01])
        # if (index==0):
        #     self.weights = np.ones((4))
        # if (index==1):
        #     self.weights = np.ones((4))
        # if (index==2):
        #     self.weights = np.ones((4))


    def pmf(self):
        return self.weights

    def attention(self, a):
        return self.pmf()[a]

    def update(self, traject, gradList):
        pass

    # def regulate(self, o):
    #     pass


class FinalPolicy:
    def __init__(self, rng, nfeatures, nactions, temp, lr_a, qWeight, lr_p, qWeightp, index, clipthres, stretchthres, stretchstep):
        self.rng = rng
        self.nfeatures = nfeatures
        self.nactions = nactions
        self.internalPI = SoftmaxPolicy(rng, nfeatures, nactions, temp, lr_p, qWeightp)
        self.attention = SigmoidAttention(nactions, lr_a, qWeight, clipthres, stretchthres, stretchstep)
        # self.attention = PredefinedAttention(index)

    def pmf(self, phi):
        pi = self.internalPI.pmf(phi)
        h = self.attention.pmf()
        normalizer = np.dot(pi, h)
        return (pi*h)/normalizer

    def sample(self, phi):
        return int(self.rng.choice(self.nactions, p=self.pmf(phi)))

    def H_update(self, traject, gradList, o):
        self.attention.update(traject, gradList)
        # print(self.attention.pmf())
        # self.attention.regulate(o)
        # self.attention.normalize()

    def P_update(self, traject, baseline=None):
        self.internalPI.update(traject, baseline)


class Distraction:
    def __init__(self, xi, R, n, h):
        self.xi = xi
        self.R = R
        self.n = n
        self.h = h

    def reward(self, r, a):
        return r - self.xi*self.R*(1-pow(self.h.attention(a),self.n))


class Objective:
    def __init__(self, weight, h):
        self.weight = weight
        self.h = h

    def grad(self):
        return None

    def loss(self):
        return None


class ValueObj(Objective):
    def __init__(self, weight, h, Qval, final):
        super().__init__(weight, h)
        self.Qval = Qval
        self.final = final

    def grad(self, phi, a):
        return self.weight * ((self.final.pmf(phi) + 1)/self.h.pmf()[a]) * self.Qval.value(phi,a)

    def loss(self):
        pass


class CoSimObj(Objective):
    hList = []

    def __init__(self, weight, h, index):
        super().__init__(weight, h)
        self.index = index

    def grad(self):
        gradient = []
        for i in range(len(self.h.pmf())):
            derivative = 0.
            exclude = 0
            for a in self.hList:
                if exclude == self.index:
                    continue
                exclude +=1

                normalizer = np.linalg.norm(self.h.pmf())*np.linalg.norm(a.pmf())
                term1 = a.pmf()[i]/normalizer
                term2 = self.h.pmf()[i]*np.dot(self.h.pmf(),a.pmf()) / (normalizer*np.power(np.linalg.norm(self.h.pmf()),2))
                derivative += -1*(term1 - term2)
            gradient.append(derivative)
        return self.weight * np.array(gradient)
    
    def loss(self):
        return np.sum([np.dot(self.h.pmf(),a.pmf())/(np.linalg.norm(self.h.pmf())*np.linalg.norm(a.pmf())) for a in self.hList])

    @classmethod
    def add2list(cls, attention):
        cls.hList.append(attention)

    @classmethod
    def reset(cls):
        cls.hList = []


class EntropyObj(Objective):
    def __init__(self, weight, h):
        super().__init__(weight, h)

    def grad(self):
        gradient = []
        normalizer = np.linalg.norm(self.h.pmf())
        normh = self.h.pmf()/normalizer
        for i in range(len(self.h.pmf())):
            term1 = (1.+np.log(normh[i]))/normalizer
            term2 = np.sum([(1.+np.log(normh[index]))*self.h.pmf()[index]/(normalizer**2) for index in range(len(self.h.pmf()))])
            # print(term2)
            gradient.append((term1-term2)*(self.loss()-0.69))
        # print(gradient)
        return self.weight * np.array(gradient)


    def loss(self):
        normalizer = np.linalg.norm(self.h.pmf())
        normh = self.h.pmf()/normalizer
        return -1*np.sum(normh * np.log(normh))


class LengthObj(Objective):
    def __init__(self, weight, h, p):
        super().__init__(weight, h)
        self.p = p

    def grad(self):
        # return 0.
        return -1 * self.weight * np.power(self.h.pmf() / self.loss(), self.p-1) * (self.loss()-1.2)
    
    def loss(self):
        return pow(np.sum(np.power(self.h.pmf(), self.p)), 1./self.p)


class Option:
    def __init__(self, rng, nfeatures, nactions, args, R, policy_over_options, index):
        self.weights = np.zeros((nfeatures, nactions)) 
        self.weightsP = np.zeros((nfeatures, nactions)) 
        self.policy = FinalPolicy(rng, nfeatures, nactions, args.temperature, args.lr_attend, self.weights, args.lr_intra, self.weightsP, index, args.clipthres, args.stretchthres, args.stretchstep)
        self.termination = SigmoidTermination(rng, nfeatures, args.lr_term, policy_over_options, args.dc)
        self.Qval = Q_U(nfeatures, nactions, args.discount, args.lr_criticA, self.weights, policy_over_options, False)
        self.PQval = Q_U(nfeatures, nactions, args.discount, args.lr_criticA_pseudo, self.weightsP, policy_over_options, True)
        self.o1 = ValueObj(args.wo1, self.policy.attention, self.Qval, self.policy)
        self.o2 = CoSimObj(args.wo2, self.policy.attention, index)
        self.o3 = EntropyObj(args.wo3, self.policy.attention)
        self.o4 = LengthObj(args.wo4, self.policy.attention, args.wo4p)
        CoSimObj.add2list(self.policy.attention)
        self.distraction = Distraction(args.xi, R, args.n, self.policy.attention)
        self.baseline = args.baseline
        self.policy_over_options = policy_over_options

    def sample(self, phi):
        return self.policy.sample(phi)

    def distract(self, reward, action):
        return self.distraction.reward(reward, action)

    def terminate(self, phi, value=False):
        if value:
            return self.termination.pmf(phi)
        else:
            return self.termination.sample(phi)

    def _Q_update(self, traject, reward, done, termination):
        self.Qval.update(traject, reward, done, termination)
        self.PQval.update(traject, self.distraction.reward(reward, traject[2]), done, termination)

    def _H_update(self, traject):
        gradList = [self.o1.grad(traject[0][0], traject[2]), self.o2.grad(), self.o3.grad(), self.o4.grad()]
        self.policy.H_update(traject, gradList, self.o4)

    def _B_update(self, phi, option):
        self.termination.update(phi, option)

    def _P_update(self, traject):
        if self.baseline:
            self.policy.P_update(traject, self.policy_over_options.value(traject[0][0], traject[0][1], pseudo=True))
        else:
            self.policy.P_update(traject)

    def update(self, traject, reward, done, phi, option, termination):
        self._Q_update(traject, reward, done, termination)
        self._H_update(traject)
        self._P_update(traject)
        self._B_update(phi, option)


class POO:
    def __init__(self, rng, nfeatures, args, R):
        self.weights = np.zeros((nfeatures, args.noptions))
        self.weightsP = np.zeros((nfeatures, args.noptions))
        self.policy = EgreedyPolicy(rng, nfeatures, args.noptions, args.epsilon, self.weights)
        self.Q_Omega = Q_O(nfeatures, args.noptions, args.discount, args.lr_critic, self.weights)
        self.Q_OmegaP = Q_O(nfeatures, args.noptions, args.discount, args.lr_critic_pseudo, self.weightsP)

    def update(self, traject, reward, distracted, done, termination):
        self.Q_Omega.update(traject, reward, done, termination)
        self.Q_OmegaP.update(traject, distracted, done, termination)

    def sample(self, phi):
        return self.policy.sample(phi)

    def advantage(self, phi, option=None):
        values = np.sum(self.weightsP[phi],axis=0)
        advantages = values - np.max(values)
        if option is None:
            return advantages
        return advantages[option]

    def value(self, phi, option=None, pseudo=False):
        if pseudo:
            if option is None:
                return np.sum(self.weightsP[phi, :], axis=0)
            return np.sum(self.weightsP[phi, option], axis=0)
        else:
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
    parser.add_argument('--lr_attend', help="Leraning rate of attention mech", type=float, default=0.5)
    parser.add_argument('--h_learn', help="Learnable attention mech", action='store_true')
    parser.add_argument('--xi', help="Exploratory", type=float, default=1.)
    parser.add_argument('--n', help="Power", type=float, default=0.5)
    parser.add_argument('--epsilon', help="Epsilon-greedy", type=float, default=1e-2)
    parser.add_argument('--nepisodes', help="Number of episodes per run", type=int, default=500)
    parser.add_argument('--nruns', help="Number of runs", type=int, default=1)
    parser.add_argument('--nsteps', help="Maximum number of steps per episode", type=int, default=2000)
    parser.add_argument('--noptions', help='Number of options', type=int, default=4)
    parser.add_argument('--baseline', help="Use the baseline for the intra-option gradient", action='store_true')
    parser.add_argument('--temperature', help="Temperature parameter for softmax", type=float, default=1e-2)
    parser.add_argument('--seed', help="seed value for experiment", type=int, default=10)
    parser.add_argument('--seed_startstate', help="seed value for starting state", type=int, default=10)
    parser.add_argument('--dc', help="deliberation cost", type=float, default=0.)
    parser.add_argument('--wo1', help="Weight of objective 1", type=float, default=0.)
    parser.add_argument('--wo2', help="Weight of objective 2", type=float, default=0.)
    parser.add_argument('--wo3', help="Weight of objective 3", type=float, default=0.)
    parser.add_argument('--wo4', help="Weight of objective 4", type=float, default=0.)
    parser.add_argument('--wo4p', help="P of objective 4", type=float, default=0.)
    parser.add_argument('--clipthres', help="Clip threshold for attention", type=float, default=0.)
    parser.add_argument('--stretchthres', help="Stretch threshold for attention", type=float, default=0.)
    parser.add_argument('--stretchstep', help="Stretch step for attention", type=float, default=0.)

    args = parser.parse_args()
    rng = np.random.RandomState(args.seed)
    env = Fourrooms(args.seed_startstate)
    screen = Display()
    R = 50.

    outer_dir = "AOAOC"
    if not os.path.exists(outer_dir):
        os.makedirs(outer_dir)

    dir_name = "Rn"+str(args.nruns)+"_Ep"+str(args.nepisodes)+ "_E"+str(args.epsilon)+"_NO"+str(args.noptions)+"_LT"+ str(args.lr_term) +\
     "_LI"+str(args.lr_intra)+"_LCA"+str(args.lr_criticA)+"_LCAP"+str(args.lr_criticA_pseudo)+"_LC"+str(args.lr_critic)+"_LCP"+str(args.lr_critic_pseudo)+"_X"+str(args.xi)+"_N"+str(args.n)+\
     "_tm"+str(args.temperature)+"_sd"+str(args.seed)

    dir_name = os.path.join(outer_dir, dir_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    save_params(args, dir_name)

    possible_next_goals = [68, 69, 70, 71, 72, 78, 79, 80, 81, 82, 88, 89, 90, 91, 92, 93, 99, 100, 101, 102, 103]

    num_states = env.observation_space.n
    num_actions = env.action_space.n

    history = np.zeros((args.nruns, args.nepisodes, 3), dtype=np.float32)

    state_frequency_history = np.zeros((args.nruns, args.nepisodes, env.observation_space.n, args.noptions),dtype=np.int32)


    for run in range(args.nruns):
        features = Tabular(env.observation_space.n)
        nfeatures, nactions = len(features), env.action_space.n

        policy_over_options = POO(rng, nfeatures, args, R)

        CoSimObj.reset()
        options = [Option(rng, nfeatures, nactions, args, R, policy_over_options, i) for i in range(args.noptions)]


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

            screen.reset(phi)

            for step in range(args.nsteps):
                observation, reward, done, _ = env.step(action)
                phi = features(observation)
                return_per_episode += pow(args.discount,step)*reward

                screen.render(phi)

                state_frequency_history[run, episode, observation, option] +=1

                last_option = option

                termination = options[option].terminate(phi, value=True)

                if options[option].terminate(phi):
                    option = policy_over_options.sample(phi)
                    option_switches += 1
                    avgduration += (1./option_switches)*(duration - avgduration)
                    duration = 1

                traject[0] = traject[1]
                traject[1] = [phi, option]
                traject[2] = action

                action = options[option].sample(phi)

                options[last_option].update(traject, reward, done, phi, last_option, termination)

                policy_over_options.update(traject, reward, options[last_option].distract(reward,traject[2]), done, termination)

                cumreward += reward
                duration += 1
                if done:
                    break

            history[run, episode, 0] = step
            history[run, episode, 1] = return_per_episode
            history[run, episode, 2] = avgduration

            print('Run {} episode {} steps {} cumreward {} avg. duration {} switches {}'.format(run, episode, step, cumreward, avgduration, option_switches))