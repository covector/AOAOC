import gym
import os
import argparse
import numpy as np
from fourrooms import Fourrooms
from scipy.special import logsumexp
from scipy.special import expit
'''
=======CLASS MAP=======
Option
    - Final Policy pi_h
        - SoftmaxPolicy(internal policy) pi_omega
        - Attention h
            - Cosine Similarity Objective o2
            - Entropy Objective o3
            - Length Objective o4
            - Distraction
    - Value Objective o1
    - Termination Func beta
    - Q_omega
    - Pseudo Q_omega
EgreedyPolicy(policy over options pi_Omega)
'''

#=======Option=======
class Option:
    def __init__(self, rng, nfeatures, nactions, args, R, policy_over_options, index):
        self.weights = np.zeros((nfeatures, nactions)) 
        self.weightsP = np.zeros((nfeatures, nactions)) 
        self.policy = FinalPolicy(rng, nfeatures, nactions, args, self.weights, self.weightsP, index)
        self.termination = SigmoidTermination(rng, nfeatures, args, policy_over_options)
        self.Qval = Q_U(nfeatures, nactions, args, self.weights, policy_over_options, False)
        self.PQval = Q_U(nfeatures, nactions, args, self.weightsP, policy_over_options, True)
        self.o1 = ValueObj(args, self.Qval, self.policy)
        self.distraction = Distraction(args, R, self.policy.attention)
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

#=======Final Policy=======
class FinalPolicy:
    def __init__(self, rng, nfeatures, nactions, args, qWeight, qWeightp, index):
        self.rng = rng
        self.nfeatures = nfeatures
        self.nactions = nactions
        self.internalPI = SoftmaxPolicy(rng, nfeatures, nactions, args, qWeightp)
        self.attention = SigmoidAttention(nactions, args, qWeight)
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

    def update(self, traject, baseline=None):
        actions_pmf = self.pmf(traject[0][0])
        critic = self.qWeight[traject[0][0], traject[2]]
        if baseline:
            critic -= baseline
        self.weights[traject[0][0], :] -= self.lr*critic*actions_pmf
        self.weights[traject[0][0], traject[2]] += self.lr*critic

#=======Attention=======
class SigmoidAttention:
    def __init__(self, nactions, args, qWeight):
        self.weights = np.random.uniform(low=-1, high=1, size=(nactions,))
        self.qWeight = qWeight
        self.lr = args.lr_attend
        self.clipthres = args.clipthres
        self.stretchthres = args.stretchthres
        self.stretchstep = args.stretchstep
        self.o2 = CoSimObj(args, index)
        self.o3 = EntropyObj(args)
        self.o4 = LengthObj(args)
        CoSimObj.add2list(self.policy.attention)

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
    #     self.weights += np.log(expo / (normalizer*(1+expo) - 1))
    #     print(np.linalg.norm(self.pmf()))
    #     if normalizer<self.clipthres:
    #         print('clip')
    #         self.weights += np.log(self.clipthres/normalizer)
    #         print(np.linalg.norm(self.pmf()))

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

#=======