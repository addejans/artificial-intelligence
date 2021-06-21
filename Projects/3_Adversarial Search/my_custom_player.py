
from sample_players import DataPlayer

# MCTS imports
from collections import defaultdict
import math, random


class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """
    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE: 
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # TODO: Replace the example implementation below with your own search
        #       method by combining techniques from lecture
        #
        # EXAMPLE: choose a random move without any search--this function MUST
        #          call self.queue.put(ACTION) at least once before time expires
        #          (the timer is automatically managed for you)
        import random
        if state.ply_count < 2:
            self.queue.put(random.choice(state.actions()))
        else:
#             self.queue.put(random.choice(state.actions()))
#             self.queue.put(self.minimax(state, depth=4))
#             self.queue.put(self.alphabeta(state, depth=4))
            self.iterative_deep_ab(state, depth_limit=4) # we don't need a depth limit since the time-out will stop us early
#             self.mcts_search(state)
    
    
    
    class MCTS:
        "Monte Carlo tree searcher. First rollout the tree then choose a move."

        def __init__(self, exploration_weight=1):
            self.Q = defaultdict(int)  # total reward of each node
            self.N = defaultdict(int)  # total visit count for each node
            self.children = dict()  # children of each node
            self.exploration_weight = exploration_weight

        def choose(self, node):
            "Choose the best successor of node. (Choose a move in the game)"
            if node.terminal_test():
                raise RuntimeError(f"choose called on terminal node {node}")
                
            if node not in self.children:
#                 return node.find_random_child()
                return random.choice(node.actions()), node.result(random.choice(node.actions())) 

            def score(n):
                if self.N[n] == 0:
                    return float("-inf")  # avoid unseen moves
                return self.Q[n] / self.N[n]  # average reward

            return None, max(self.children[node], key=score)

        def do_rollout(self, node):
            "Make the tree one layer better. (Train for one iteration.)"
            path = self._select(node)
            leaf = path[-1]
            self._expand(leaf)
            reward = self._simulate(leaf)
            self._backpropagate(path, reward)

        def _select(self, node):
            "Find an unexplored descendent of `node`"
            path = []
            while True:
                path.append(node)
                if node not in self.children or not self.children[node]:
                    # node is either unexplored or terminal
                    return path
                unexplored = self.children[node] - self.children.keys()
                if unexplored:
                    n = unexplored.pop()
                    path.append(n)
                    return path
                node = self._uct_select(node)  # descend a layer deeper

        def _expand(self, node):
            "Update the `children` dict with the children of `node`"
            if node in self.children:
                return  # already expanded
#             self.children[node] = node.find_children()
                self.children[node] = node.actions() #node.liberties()

        def _simulate(self, node):
            "Returns the reward for a random simulation (to completion) of `node`"
            invert_reward = True
            while True:
                if node.terminal_test():
#                     reward = node.reward()
#                     reward = node.utility(self.player_id)
                    reward = node.utility(0) # TODO: Figure this out
                    return 1 - reward if invert_reward else reward
#                 node = node.find_random_child()
                node = node.result(random.choice(node.actions()))
                invert_reward = not invert_reward

        def _backpropagate(self, path, reward):
            "Send the reward back up to the ancestors of the leaf"
            for node in reversed(path):
                self.N[node] += 1
                self.Q[node] += reward
                reward = 1 - reward  # 1 for me is 0 for my enemy, and vice versa

        def _uct_select(self, node):
            "Select a child of node, balancing exploration & exploitation"

            # All children of node should already be expanded:
            assert all(n in self.children for n in self.children[node])

            log_N_vertex = math.log(self.N[node])

            def uct(n):
                "Upper confidence bound for trees"
                return self.Q[n] / self.N[n] + self.exploration_weight * math.sqrt(
                    log_N_vertex / self.N[n]
                )

            return max(self.children[node], key=uct)
        
    def mcts_search(self, state):
        tree = self.MCTS()
        while True:
            if state.terminal_test(): return state.utility(self.player_id)
        
            # You can train as you go, or only at the beginning.
            # Here, we train as we go, doing fifty rollouts each turn.
            # TODO: ^^^ What does this mean?
            for _ in range(50): # Default was 50: https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1
                tree.do_rollout(state)
            action, state = tree.choose(state)
            # add the action that got us to this chosen state!
            # TODO: I don't think we want to put this in the queue. 
            #       contunually adding to this queue doesn't make sense. You are updating the state each time and thus the last move placed
            #       in the queue won't even be feasible! 
            self.queue.put(action) # TODO: Need to pass actions back from MCTS.. populate the queue with best from each iteration as we go?
            break #added this because of the TODO listed above self.queue.put(action)
            if state.terminal_test():
                break
                                
    def iterative_deep_ab(self, state, depth_limit):
        best_move = None
        for depth in range(1, depth_limit+1):
            self.queue.put(self.alphabeta(state, depth))

    
    def alphabeta(self, state, depth):
        
        def min_value(state, depth, alpha, beta):
            if state.terminal_test(): 
                return state.utility(self.player_id)
            if depth <= 0: return self.score(state)
            value = float("inf")
            for action in state.actions():
                value = min(value, max_value(state.result(action), depth - 1, alpha, beta))
                if value <= alpha:
                    return value
                beta = min(beta, value)
            return value


        def max_value(state, depth, alpha, beta):
            if state.terminal_test(): return state.utility(self.player_id)
            if depth <= 0: return self.score(state)
            value = float("-inf")
            for action in state.actions():
                value = max(value, min_value(state.result(action), depth - 1, alpha, beta))
                if value >= beta:
                    return value
                alpha = max(alpha, value)
            return value

        alpha = float("-inf")
        beta = float("inf")
        best_score = float("-inf")
        best_move = None
        for action in state.actions():
            value = min_value(state.result(action), depth - 1, alpha, beta)
            alpha = max(alpha, value)
            if value >= best_score:
                best_score = value
                best_move = action
        return best_move
        
        
    def minimax(self, state, depth):

        def min_value(state, depth):
            if state.terminal_test(): return state.utility(self.player_id)
            if depth <= 0: return self.score(state)
            value = float("inf")
            for action in state.actions():
                value = min(value, max_value(state.result(action), depth - 1))
            return value

        def max_value(state, depth):
            if state.terminal_test(): return state.utility(self.player_id)
            if depth <= 0: return self.score(state)
            value = float("-inf")
            for action in state.actions():
                value = max(value, min_value(state.result(action), depth - 1))
            return value

        return max(state.actions(), key=lambda x: min_value(state.result(x), depth - 1))
    
    
    def score(self, state):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        return len(own_liberties) - len(opp_liberties)
