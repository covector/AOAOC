import numpy as np
from fourrooms import Fourrooms
from scipy.special import logsumexp, expit, softmax
'''
=======CLASS MAP=======
Option
    - FinalPolicy pi_h
        - Internal Policy (SoftmaxPolicy) pi_omega
        - Attention Unit (LearnableAttention/PredefinedAttention) h_omega
            - Value Objective (ValueObj) o1
            - Cosine Similarity Objective (CoSimObj) o2
            - Entropy Objective (EntropyObj) o3
            - Length Objective (LengthObj) o4
    - Termination Function (SigmoidTermination) beta_omega
    - Q_omega (Q_O)
Policy Over Options (POO) pi_Omega
    - Policy (EgreedyPolicy)
    - Q_Omega (Q_U)
'''
#=======Option=======
class Option:
    def __init__(self, rng, nfeatures, nactions, args, policy_over_options, index):
        self.weights = np.zeros((nfeatures, nactions)) 
        self.policy = FinalPolicy(rng, nfeatures, nactions, args, self.weights, index)
        self.termination = SigmoidTermination(rng, nfeatures, args)
        self.Qval = Q_U(nfeatures, nactions, args, self.weights, policy_over_options)

    def sample(self, phi):
        return self.policy.sample(phi)

    def terminate(self, phi, value=False):
        if value:
            return self.termination.pmf(phi)
        else:
            return self.termination.sample(phi)

    def _Q_update(self, traject, reward, done, termination):
        self.Qval.update(traject, reward, done, termination)

    def _H_update(self, traject):
        qVal = self.Qval.value(traject[0][0], traject[2])
        self.policy.H_update(traject, qVal)

    def _B_update(self, phi, option, advantage):
        self.termination.update(phi, option, advantage)

    def _P_update(self, traject, baseline):
        self.policy.P_update(traject, baseline)

    def update(self, traject, reward, done, phi, option, termination, baseline, advantage):
        self._Q_update(traject, reward, done, termination)
        self._H_update(traject)
        self._P_update(traject, baseline)
        self._B_update(phi, option, advantage)


#=======Final Policy=======
class FinalPolicy:
    def __init__(self, rng, nfeatures, nactions, args, qWeight, index):
        self.rng = rng
        self.nactions = nactions
        self.internalPI = SoftmaxPolicy(rng, nfeatures, nactions, args, qWeight)
        if (args.h_learn):
            self.attention = LearnableAttention(nactions, args, index)
        else:
            self.attention = PredefinedAttention(args, index)

    def pmf(self, phi):
        pi = self.internalPI.pmf(phi)
        h = self.attention.pmf()
        normalizer = np.dot(pi, h)
        return (pi*h)/normalizer

    def sample(self, phi):
        return int(self.rng.choice(self.nactions, p=self.pmf(phi)))

    def H_update(self, traject, qVal):
        self.attention.update(traject, self.pmf(traject[0][0]), qVal)
        
    def P_update(self, traject, baseline):
        self.internalPI.update(traject, baseline)


#=======Internal Policy=======
class SoftmaxPolicy:
    def __init__(self, rng, nfeatures, nactions, args, qWeight):
        self.rng = rng
        self.nactions = nactions
        self.temp = args.temp
        self.weights = np.zeros((nfeatures, nactions))
        self.qWeight = qWeight
        self.lr = args.lr_intra

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
class LearnableAttention():
    def __init__(self, nactions, args, index):
        self.weights = np.random.uniform(low=-1, high=1, size=(nactions,))
        self.lr = args.lr_attend
        self.o1 = ValueObj(args)
        self.o2 = CoSimObj(args, index)
        self.o3 = EntropyObj(args)
        self.o4 = LengthObj(args)
        CoSimObj.add2list(self)
        self.normalize = args.normalize

    def pmf(self):
        if self.normalize:
            return np.clip(softmax(self.weights), 0.05, None)
        return expit(self.weights)

    def _grad(self):
        attend = self.pmf()
        return attend*(1. - attend)

    def attention(self, a):
        return self.pmf()[a]

    def update(self, traject, finalPmf, qVal):
        hPmf = self.pmf()
        gradList = [self.o1.grad(traject[0][0], traject[2], hPmf, finalPmf, qVal), self.o2.grad(hPmf), self.o3.grad(hPmf), self.o4.grad(hPmf)]
        self.weights += self.lr * np.sum(gradList, axis=0) * self._grad()
        if self.normalize:
            self.normalizing()

    def normalizing(self):
        self.weights -= np.mean(self.weights)


class PredefinedAttention():
    def __init__(self, args, index):
        if (index==0):
            self.weights = np.array([1, 1, 1, 1])
        if (index==1):
            self.weights = np.array([1, 1, 1, 1])

    def pmf(self):
        return self.weights

    def attention(self, a):
        return self.pmf()[a]

    def update(self, traject, finalPmf, qVal):
        pass


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

    def grad(self, phi, a, hPmf, finalPmf, qVal):   # proof is in appedix B.1
        return self.weight * ((finalPmf + 1)/hPmf[a]) * qVal

    def loss(self):
        pass


class CoSimObj(Objective):
    hList = []

    def __init__(self, args, index):
        super().__init__(args.wo2)
        self.index = index

    def grad(self, hPmf):   # proof is in appedix B.2
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

    def grad(self, hPmf):   # proof is in appedix B.3
        gradient = []
        normalizer = np.sum(hPmf)
        normh = hPmf/normalizer
        for i in range(len(hPmf)):
            term1 = (1.+np.log(normh[i]))/normalizer
            term2 = np.sum([(1.+np.log(normh[index]))*hPmf[index]/(normalizer**2) for index in range(len(hPmf))])
            gradient.append((term1-term2)*(self.loss(hPmf)-0.69))
        return self.weight * np.array(gradient)

    def loss(self, hPmf):
        normalizer = np.sum(hPmf)
        normh = hPmf/normalizer
        return -1*np.sum(normh * np.log(normh))


class LengthObj(Objective):
    def __init__(self, args):
        super().__init__(args.wo4)

    def grad(self, hPmf):   # proof is in appedix B.4
        return -1 * hPmf / self.loss(hPmf)
    

    def loss(self, hPmf):
        return np.linalg.norm(hPmf)


#=======Termination Function=======
class SigmoidTermination:
    def __init__(self, rng, nfeatures, args):
        self.rng = rng
        self.weights = np.zeros((nfeatures,))
        self.lr = args.lr_term
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
        self.weights[direction] -= self.lr*magnitude*(advantage+self.dc)


#=======Q-Value Individual Option=======
class Q_U:
    def __init__(self, nfeatures, nactions, args, weights, policy_over_options):
        self.weights = weights
        self.lr = args.lr_criticA
        self.discount = args.discount
        self.policy_over_options = policy_over_options
    
    def value(self, phi, action):
        return np.sum(self.weights[phi, action], axis=0)

    def update(self, traject, reward, done, termination):
        update_target = reward
        if not done:
            current_values = self.policy_over_options.value(traject[1][0])
            update_target += self.discount*((1. - termination)*current_values[traject[0][1]] + termination*np.max(current_values))

        tderror = update_target - self.value(traject[0][0], traject[2])
        self.weights[traject[0][0], traject[2]] += self.lr*tderror


#=======Policy Over Option=======
class POO:
    def __init__(self, rng, nfeatures, args):
        self.weights = np.zeros((nfeatures, args.noptions))
        self.policy = EgreedyPolicy(rng, args, self.weights)
        self.Q_Omega = Q_O(args, self.weights)

    def update(self, traject, reward, done, termination):
        self.Q_Omega.update(traject, reward, done, termination)

    def sample(self, phi):
        return self.policy.sample(phi)

    def advantage(self, phi, option=None):
        values = np.sum(self.weights[phi],axis=0)
        advantages = values - np.max(values)
        if option is None:
            return advantages
        return advantages[option]

    def value(self, phi, option=None):
        if option is None:
            return np.sum(self.weights[phi, :], axis=0)
        return np.sum(self.weights[phi, option], axis=0)


class EgreedyPolicy:
    def __init__(self, rng, args, weights):
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
    def __init__(self, args, weights):
        self.weights = weights
        self.lr = args.lr_critic
        self.discount = args.discount

    def _value(self, phi, option=None):
        if option is None:
            return np.sum(self.weights[phi, :], axis=0)
        return np.sum(self.weights[phi, option], axis=0)

    def update(self, traject, reward, done, termination):
        update_target = reward
        if not done:
            current_values = self._value(traject[1][0])
            update_target += self.discount*((1. - termination)*current_values[traject[0][1]] + termination*np.max(current_values))

        tderror = update_target - self._value(traject[0][0], traject[0][1])
        self.weights[traject[0][0],traject[0][1]] += self.lr*tderror


#=======Standard=======
# Follow the code standard of the ioc repository
class Tabular:
    def __init__(self, nstates):
        self.nstates = nstates

    def __call__(self, state):
        return np.array([state,])

    def __len__(self):
        return self.nstates