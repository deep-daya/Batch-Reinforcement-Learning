import collections
import pandas as pd
import sys
import cvxpy
class MDP:
    def __init__(self,S=0,A=0,T={},R={},statesUsed = 0, keysT = {},type="",data=[]):
        self.S = S
        self.A = A
        self.T = T
        self.R = R
        self.statesUsed = statesUsed
        self.keysT = keysT
        self.data= data
        if (type == "small") | (type == "large"):
            self.gamma = 0.95
        else:
            self.gamma = 1
        self.type = type
        self.__doc__ = "This class stores a MDP for a given State Space S, Action Space A, and Reward, Transition p" \
                       "probability space. We can find a policy using our preferred method in the write policy function."

    def writepolicy(self,filename_in, filename_out):
    	"""Executes the Q-learning Program for the large and medium files while using value function optimization for the small file.
    	Saves the policy to filename_out. """
        policy = {}
        if self.type == "small":
            vp = ValueFunctionPolicy(self, [])
            policy = vp.optimization_small()
        else:
            ql = Q_Learning(self, alpha = 0.05, iterations=30)
            print("here")
            policy = ql.computePolicy()
        with open(filename_out, 'w') as f:
            for value in policy.values():
                f.write("{0}\n".format(value))

class Q_Learning:
    def __init__(self, MDP,alpha = 0.1, iterations = 100):
        self.Q = collections.defaultdict(int)
        self.alpha = alpha
        self.MDP = MDP
        self.iterations = iterations
        self.data = MDP.data
        self.absStates = list(set(self.data["s"]) ^ set(self.data["sp"]))
        self.__doc__ = "This class performs Q_learning on a Problem MDP for a given Alpha and number of iterations."


    def updateQ(self):
    	"""Uses the Q learning update in order to update the Q function."""
        Q = self.Q
        delta = 0
        it = 0
        print(len(self.data["s"].unique()))
        for i in self.data["s"].unique():
            if it%100 == 0: print(it)
            it+=1
            s = i
            if s in self.absStates:
                continue
            possibleActions = self.data.loc[self.data["s"] == s]["a"].unique()
            for a in possibleActions:
                biggest_update = Q[(s,a)]
                reward, sp = self.find_max_next_state(s,a)
                if sp in self.absStates:
                    maxQS = 0
                else:
                    maxQS = self.maxQ(sp)
                Q[(s,a)] = self.alpha*(reward + self.MDP.gamma*maxQS - Q[(s, a)])
                biggest_update = Q[(s,a)] - biggest_update
            if biggest_update > delta:
                delta = biggest_update
        self.Q = Q
        return delta

    def computePolicy(self):
    	"""Computes the policy by applying the Q-learning update multiple times and then figures out the policy from the Q function."""
        convergence = 10
        threshold = 0.0001
        ite = 1
        while (ite < self.iterations) & (convergence > threshold):
            print("Episode {0}".format(ite))
            convergence = self.updateQ()
            print(convergence)
            ite += 1
        print("Convergence reached. Initializing Policy")
        pi = {}
        for i in range(self.MDP.S):
            res = {}
            for j in range(self.MDP.A):
                res[j+1] = self.Q[(i+1,j+1)]

            pi[i+1] = max(res, key=lambda key: res[key])
        if (self.MDP.type == "medium"):
            pi = self.nearby_states(pi)
        elif self.MDP.type == "large":
            pi =  self.same_action(pi)
        return pi


    def nearby_states(self, policy):
    	"""Helper function used the find nearby states for the medium data file since the domain is discretized from a continuous one."""
        statesUsed = list(set(self.data["s"].unique()).union(set(self.data["sp"])))
        for i in range(self.MDP.S):
            if i+1 not in statesUsed:
                k = i+1
                lst = statesUsed
                policy[i+1] = policy[lst[min(range(len(lst)), key=lambda i: abs(lst[i] - k))]]
        return policy

    def same_action(self,policy):
    	"""Helper Function to assign unknown policy steps to the closest known action in the policy."""
        statesUsed = list(set(self.data["s"].unique()).union(set(self.data["sp"])))
        for i in range(self.MDP.S):
            if i+1 not in statesUsed:
                policy[i+1] = 9
        return policy
    # def big_brain_time(self, policy):
    #     statesUsed = self.data["s"].unique()
    #     for i in range(self.MDP.S):
    #         if i+1 not in statesUsed:
    #             if i+1 > 16000:
    #                 if (i+1)%1000 == 1:
    #                     policy[i+1] = 1
    #                 elif (i+1)%100 == 2:
    #
    #     return policy




    def find_max_next_state(self,s,a):
    	"""Helper function to Max reward step out of all possible next actions."""
        max_next_action_reward = self.data.loc[self.data["s"]==s].loc[self.data["a"]==a].groupby("s").max("r")
        reward = max_next_action_reward["r"].values[0]
        sp = max_next_action_reward["sp"].values[0]
        return reward, sp

    def maxQ(self, sdash):
    	"""Helper Function to find the Max Q of the next action out of all possible next actions."""
        Qdash = []
        possibleNextActions = self.data.loc[self.data["s"] == sdash]["a"].unique()
        for i in possibleNextActions:
            Qdash.append(self.Q[(sdash,i)])
        return max(Qdash)



class ValueFunctionPolicy:
    def __init__(self,MDP, U):
        self.MDP = MDP
        self.U = U
        self.__doc__  = "This class provides the ability to find a policy using Value Function Optimization"

    def optimization_small(self):
    	"""Formulates an optimization to minimize the sum of the Value Function using the MDP's Transition Function, Reward Function and all possible actions"""
        U = cvxpy.Variable(self.MDP.S)
        constraints = []
        for i in range(self.MDP.S):
            for a in range(self.MDP.A):
                keys = [(i+1,a+1,j) for j in self.MDP.keysT[i+1][a+1]]
                constraints += [U[i] >= self.MDP.R[(i+1,a+1)] + self.MDP.gamma*sum([self.MDP.T[key] * U[key[2]-1] for key in keys])]
        problem = cvxpy.Problem(cvxpy.Minimize(sum(U)),constraints)
        problem.solve(solver="ECOS")
        U = U.value
        self.U = U
        return self.greedy_policy_finder()

    def greedy_policy_finder(self):
    	"""Using the resulting U from optimization_small function, this function finds the policy which gives the resulting U."""
        pi = {}
        for i in range(self.MDP.S):
            res = {}
            for j in range(self.MDP.A):
                res[j+1] = (self.lookahead(i+1,j+1))
            pi[i+1] = max(res, key=lambda key: res[key])
        return pi

    def lookahead(self, s, a):
    	"""This function provides the value function approximation given a current state and action."""
        S,T,R,gamma,keysT = self.MDP.S, self.MDP.T, self.MDP.R, self.MDP.gamma, self.MDP.keysT
        keys = [(s,a,j) for j in keysT[s][a]]
        return R[(s,a)] + gamma * sum([T[key]*self.U[key[2]-1] for key in keys])

class Problem():
    def __init__(self,filename, filename_out,types=""):
        self.filename = filename
        self.filename_out = filename_out
        self.type = types
        self.__doc__ = "Class which solves the three problems and establishes the basic MDP for all problems."

    def read_data(self):
        data = pd.read_csv(self.filename)
        return data

    def compute_policy_and_parsing_func(self):
    	"""Establishes the MDP for each problem."""
        data = self.read_data()
        numberStates = max(data["s"].unique())
        if (self.type == "small") & (numberStates != 100):
            numberStates = 100
        elif (self.type == "medium") & (numberStates!= 50000):
            numberStates = 50000
        else:
            numberStates = 312020

        numberActions = max(data["a"].unique())
        statesUsed = len(data["s"].unique())
        rewards = collections.defaultdict(int)
        transition = collections.defaultdict(int)
        s_a_sdash = collections.defaultdict(int)
        if self.type == "small":
            for i in range(numberStates):
                ind_state = data.loc[data["s"] == i + 1].index.values
                dict_a_sdash = data.iloc[ind_state].groupby(["a", "sp"]).groups
                n_sa = {}
                for action in range(numberActions):
                    ind_action = data.loc[ind_state].loc[data["a"] == action + 1].index.values
                    n_sa[action + 1] = len(ind_action)
                    rewards[(i + 1, action + 1)] = data.loc[ind_state].loc[ind_action]["r"].max()
                for j in dict_a_sdash.keys():
                    if n_sa[j[0]] == 0:
                        transition[(i + 1, j[0], j[1])] = 0
                    else:
                        transition[(i + 1, j[0], j[1])] = len(dict_a_sdash[j]) / n_sa[j[0]]
                    if i + 1 in s_a_sdash.keys():
                        if j[0] in s_a_sdash[i + 1].keys():
                            s_a_sdash[i + 1][j[0]].append(j[1])
                        else:
                            s_a_sdash[i + 1][j[0]] = [j[1]]
                    else:
                        s_a_sdash[i + 1] = collections.defaultdict(int, {j[0]: [j[1]]})
            return MDP(S=numberStates,A=numberActions,T=transition,R=rewards,
                   statesUsed = statesUsed, keysT = s_a_sdash,type="small",data=data).writepolicy(self.filename, self.filename_out)
        else:
            return MDP(S=numberStates,A=numberActions,T=transition,R=rewards,
                   statesUsed = statesUsed, keysT = s_a_sdash,type="medium",data=data).writepolicy(self.filename, self.filename_out)

def main():
    inputfile = sys.argv[1]
    outputfile = sys.argv[2]
    type = sys.argv[3]
    print(type)
    prob = Problem(inputfile, outputfile, type)
    prob.compute_policy_and_parsing_func()

if __name__ == '__main__':
    main()
