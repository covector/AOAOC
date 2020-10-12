'''
This code is based on the ioc repository [13]
'''

import numpy as np
from fourrooms import Fourrooms
from scipy.special import logsumexp
from scipy.special import expit

'''
=======CLASS MAP=======
Option
    - FinalPolicy pi_h
        - Internal Policy (SoftmaxPolicy) pi_omega
        - Attention Unit (SigmoidAttention/PredefinedAttention) h_omega
            - Value Objective (ValueObj) o1
            - Cosine Similarity Objective (CoSimObj) o2
            - Entropy Objective (EntropyObj) o3
            - Length Objective (LengthObj) o4
    - Termination Function (SigmoidTermination) beta_omega
    - Q_omega (Q_O)
    - Pseudo Q_omega (Q_O)
Policy Over Options (POO) pi_Omega
    - Policy (EgreedyPolicy)
    - Q_Omega (Q_U)
    - Pseudo Q_Omega (Q_U)
'''

#=======Option=======
class Option:
    def __init__(self, rng, nfeatures, nactions, args, R, policy_over_options, index):
        self.weights = np.zeros((nfeatures, nactions)) 
        self.weightsP = np.zeros((nfeatures, nactions)) 
        self.policy = FinalPolicy(rng, nfeatures, nactions, args, R, self.weights, self.weightsP, index)
        self.termination = SigmoidTermination(rng, nfeatures, args)
        self.Qval = Q_U(nfeatures, nactions, args, self.weights, policy_over_options, False)
        self.PQval = Q_U(nfeatures, nactions, args, self.weightsP, policy_over_options, True)

    def sample(self, phi):
        return self.policy.sample(phi)

    def distract(self, reward, action):
        return self.policy.distract(reward, action)

    def terminate(self, phi, value=False):
        if value:
            return self.termination.pmf(phi)
        else:
            return self.termination.sample(phi)

    def _Q_update(self, traject, reward, done, termination):
        self.Qval.update(traject, reward, done, termination)
        self.PQval.update(traject, self.distraction.reward(reward, traject[2]), done, termination)

    def _H_update(self, traject):
        self.policy.H_update(traject)

    def _B_update(self, phi, option):
        self.termination.update(phi, option)

    def _P_update(self, traject, baseline):
        # self.policy.P_update(traject, self.policy_over_options.value(traject[0][0], traject[0][1], pseudo=True))
        self.policy.P_update(traject, baseline)

    def update(self, traject, reward, done, phi, option, termination, baseline):
        self._Q_update(traject, reward, done, termination)
        self._H_update(traject)
        self._P_update(traject, baseline)
        self._B_update(phi, option)

#=======Final Policy=======
class FinalPolicy:
    def __init__(self, rng, nfeatures, nactions, args, R, qWeight, qWeightp, index):
        self.rng = rng
        self.nactions = nactions
        self.internalPI = SoftmaxPolicy(rng, nfeatures, nactions, args, qWeightp)
        #=======predefine attention?========
        self.attention = SigmoidAttention(nactions, args, R, qWeight)
        # self.attention = PredefinedAttention(index)
        #===================================

    def pmf(self, phi):
        pi = self.internalPI.pmf(phi)
        h = self.attention.pmf()
        normalizer = np.dot(pi, h)
        return (pi*h)/normalizer

    def sample(self, phi):
        return int(self.rng.choice(self.nactions, p=self.pmf(phi)))

    def H_update(self, traject):
        self.attention.update(traject, self.pmf(traject[0][0]))
        # print(self.attention.pmf())
        # self.attention.regulate(o)
        # self.attention.normalize()

    def P_update(self, traject, baseline):
        self.internalPI.update(traject, baseline)

    def distract(self, reward, action):
        self.attention.distract(reward, action)

#=======Internal Policy=======
class SoftmaxPolicy:
    def __init__(self, rng, nfeatures, nactions, args, qWeight):
        self.rng = rng
        self.nactions = nactions
        self.temp = args.temp
        self.weights = np.zeros((nfeatures, nactions))
        self.qWeight = qWeight
        self.lr =args.lr_intra

    def _value(self, phi, action=None):
        if action is None:
            return np.sum(self.weights[phi, :], axis=0)
        return np.sum(self.weights[phi, action], axis=0)

    def pmf(self, phi):
        v = self._value(phi)/self.temp
        return np.exp(v - logsumexp(v))

    def sample(self, phi):
        return int(self.rng.choice(self.nactions, p=self.pmf(phi)))

    def update(self, traject, baseline):
        actions_pmf = self.pmf(traject[0][0])
        critic = self.qWeight[traject[0][0], traject[2]]
        if baseline:
            critic -= baseline
        self.weights[traject[0][0], :] -= self.lr*critic*actions_pmf
        self.weights[traject[0][0], traject[2]] += self.lr*critic

#=======Attention=======
class Attention:
    def __init__(self, args):
        self.xi = args.xi
        self.R = R
        self.n = args.n

    def pmf(self):
        return None

    def attention(self, a):
        return self.pmf()[a]

    def update(self, traject, gradList):
        pass

    def distract(self, reward, action):
        return reward - self.xi*self.R*(1-pow(self.attention(action),self.n))


class SigmoidAttention(Attention):
    def __init__(self, nactions, args, R, qWeight):
        super().__init__(self, args)
        self.weights = np.random.uniform(low=-1, high=1, size=(nactions,))
        self.qWeight = qWeight
        self.lr = args.lr_attend
        self.clipthres = args.clipthres
        self.stretchthres = args.stretchthres
        self.stretchstep = args.stretchstep
        self.o1 = ValueObj(args)
        self.o2 = CoSimObj(args, index)
        self.o3 = EntropyObj(args)
        self.o4 = LengthObj(args)
        CoSimObj.add2list(self)

    def pmf(self):
        return expit(self.weights)

    def _grad(self):
        attend = self.pmf()
        return attend*(1. - attend)

    def attention(self, a):
        return self.pmf()[a]

    def update(self, traject, finalPmf):
        hPmf = self.pmf()
        gradList = [self.o1.grad(traject[0][0], traject[2], hPmf, finalPmf), self.o2.grad(hPmf), self.o3.grad(hPmf), self.o4.grad(hPmf)]
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
    #     self.weights += np.log(expo / (normalizer*(1+expo) - 1))
    #     print(np.linalg.norm(self.pmf()))
    #     if normalizer<self.clipthres:
    #         print('clip')
    #         self.weights += np.log(self.clipthres/normalizer)
    #         print(np.linalg.norm(self.pmf()))


class PredefinedAttention(Attention):
    def __init__(self, args, index):
        super().__init__(self, args)
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

#=======Objectives========
class Objective:
    def __init__(self, weight):
        self.weight = weight

    def grad(self):
        return None

    def loss(self):
        return None


class ValueObj(Objective):
    def __init__(self, args):
        super().__init__(args.wo1)
        self.Qval = Qval
        self.final = final

    def grad(self, phi, a, hPmf, finalPmf):
        return self.weight * ((finalPmf + 1)/hPmf[a]) * self.Qval.value(phi,a)

    def loss(self):
        pass


class CoSimObj(Objective):
    hList = []

    def __init__(self, args, index):
        super().__init__(args.wo2)
        self.index = index

    def grad(self, hPmf):
        gradient = []
        for i in range(len(hPmf)):
            derivative = 0.
            exclude = 0
            for a in self.hList:
                if exclude == self.index:
                    continue
                exclude +=1

                normalizer = np.linalg.norm(hPmf)*np.linalg.norm(a.pmf())
                term1 = a.pmf()[i]/normalizer
                term2 = hPmf[i]*np.dot(hPmf,a.pmf()) / (normalizer*np.power(np.linalg.norm(hPmf),2))
                derivative += -1*(term1 - term2)
            gradient.append(derivative)
        return self.weight * np.array(gradient)
    
    def loss(self):
        return np.sum([np.dot(hPmf,a.pmf())/(np.linalg.norm(hPmf)*np.linalg.norm(a.pmf())) for a in self.hList])

    @classmethod
    def add2list(cls, attention):
        cls.hList.append(attention)

    @classmethod
    def reset(cls):
        cls.hList = []


class EntropyObj(Objective):
    def __init__(self, args):
        super().__init__(args.wo3)

    def grad(self, hPmf):
        gradient = []
        normalizer = np.linalg.norm(hPmf)
        normh = hPmf/normalizer
        for i in range(len(hPmf)):
            term1 = (1.+np.log(normh[i]))/normalizer
            term2 = np.sum([(1.+np.log(normh[index]))*hPmf[index]/(normalizer**2) for index in range(len(hPmf))])
            # print(term2)
            gradient.append((term1-term2)*(self.loss()-0.69))
        # print(gradient)
        return self.weight * np.array(gradient)

    def loss(self):
        normalizer = np.linalg.norm(hPmf)
        normh = hPmf/normalizer
        return -1*np.sum(normh * np.log(normh))


class LengthObj(Objective):
    def __init__(self, args):
        super().__init__(args.wo4)
        self.p = args.wo4p

    def grad(self, hPmf):
        # return 0.
        return -1 * self.weight * np.power(hPmf / self.loss(), self.p-1) * (self.loss()-1.2)
    
    def loss(self, hPmf):
        return pow(np.sum(np.power(hPmf, self.p)), 1./self.p)

#=======Termination Function=======
class SigmoidTermination:
    def __init__(self, rng, nfeatures, args):
        self.rng = rng
        self.weights = np.zeros((nfeatures,))
        self.lr = args.lr
        self.dc = args.dc

    def pmf(self, phi):
        return expit(np.sum(self.weights[phi]))

    def sample(self, phi):
        return int(self.rng.uniform() < self.pmf(phi))

    def _grad(self, phi):
        terminate = self.pmf(phi)
        return terminate*(1. - terminate), phi
    
    def update(self, phi, option, advantage):
        magnitude, direction = self._grad(phi)
        # self.weights[direction] -= self.lr*magnitude*(self.policy_over_options.advantage(phi, option)+self.dc)
        self.weights[direction] -= self.lr*magnitude*(advantage+self.dc)

#=======Q-Value Individual Option=======
class Q_U:
    def __init__(self, nfeatures, nactions, args, weights, policy_over_options, pseudo):
        self.weights = weights
        if pseudo:
            self.lr = args.lr_criticA_pseudo
        else:
            self.lr = args.lr_criticA
        self.discount = args.discount
        self.policy_over_options = policy_over_options
        self.pseudo = pseudo
    
    def value(self, phi, action):
        return np.sum(self.weights[phi, action], axis=0)

    def update(self, traject, reward, done, termination, value):
        # One-step update target
        update_target = reward
        if not done:
            current_values = self.policy_over_options.value(traject[1][0], pseudo=self.pseudo)
            update_target += self.discount*((1. - termination)*current_values[traject[0][1]] + termination*np.max(current_values))

        # Update values upon arrival if desired
        tderror = update_target - self.value(traject[0][0], traject[2])
        self.weights[traject[0][0], traject[2]] += self.lr*tderror

#=======Policy Over Option=======
class POO:
    def __init__(self, rng, nfeatures, args):
        self.weights = np.zeros((nfeatures, args.noptions))
        self.weightsP = np.zeros((nfeatures, args.noptions))
        self.policy = EgreedyPolicy(rng, nfeatures, args, self.weights)
        self.Q_Omega = Q_O(nfeatures, self.weights, False)
        self.Q_OmegaP = Q_O(nfeatures, self.weightsP, True)

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

class EgreedyPolicy:
    def __init__(self, rng, nfeatures, args, weights):
        self.rng = rng
        self.epsilon = args.epsilon
        self.noptions = args.noptions
        self.weights = weights

    def _value(self, phi, action=None):
        if action is None:
            return np.sum(self.weights[phi, :], axis=0)
        return np.sum(self.weights[phi, action], axis=0)

    def sample(self, phi):
        if self.rng.uniform() < self.epsilon:
            return int(self.rng.randint(self.weights.shape[1]))
        return int(np.argmax(self._value(phi)))

#=======Q-Value All Option=======
class Q_O:
    def __init__(self, args, weights, pseudo):
        self.weights = weights
        if pseudo:
            self.lr = args.lr_critic_pseudo
        else:
            self.lr = args.lr_critic
        self.discount = args.discount

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

#=======Standard=======
#Follow the code standard of the ioc repository
class Tabular:
    def __init__(self, nstates):
        self.nstates = nstates

    def __call__(self, state):
        return np.array([state,])

    def __len__(self):
        return self.nstates

#Replace the command line argparse
class Arguments:
    def __init__(self):
            self.discount=0.99
            self.lr_term=0.1
            self.lr_intra=0.25
            self.lr_critic=0.5
            self.lr_critic_pseudo=0.5
            self.lr_criticA=0.5
            self.lr_criticA_pseudo=0.5
            self.lr_attend=0.02
            self.h_learn=False
            self.xi=1.
            self.n=0.5
            self.epsilon=1e-1
            self.nepisodes=4000
            self.nruns=1
            self.nsteps=2000
            self.noptions=3
            self.baseline=True
            self.temperature=1.
            self.seed=2222
            self.seed_startstate=1111
            self.dc = 0.1
            self.wo1 = 1.   #q
            self.wo2 = 2.    #cosim
            self.wo3 = 2.    #entropy
            self.wo4 = 5.    #size
            self.wo4p = 2
            self.clipthres = 0.1
            self.stretchthres = 1.
            self.stretchstep = 1.