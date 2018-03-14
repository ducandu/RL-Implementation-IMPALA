"""
 -------------------------------------------------------------------------
 IMPALA_proto - 
 random_env.py
 
 !!TODO: add file description here!! 
  
 created: 2018/03/09 in PyCharm
 (c) 2017 Sven - ducandu GmbH
 -------------------------------------------------------------------------
"""


class Env(object):
    """
    Simple 3-state env with reversed 2nd state from Barto and Sutton book draft 2017 Chapter 13 (policy gradient methods)
    [0 1 2 G]
    start state==0
    actions: left, right (deterministic)
    only in state==1, actions have reverse effect
    observation does not include state
    optimal policy would be stochastic (0.59 right, 0.41 left)
    """
    def __init__(self):
        self.num_actions = 2  # 0=left and 1=right
        self.state = 0  # the current state

    def execute(self, action):
        reward = -1
        is_terminal = False
        if self.state != 1:
            if action == 0:
                self.state -= 1
                if self.state < 0:
                    self.state = 0
            else:
                self.state += 1
                if self.state == 3:
                    reward = 1
                    is_terminal = True
        # the "crazy" state
        elif self.state == 1:
            if action == 1:
                self.state = 0
            else:
                self.state = 2
        # return only reward, is-terminal (no state observation; we are blind)
        return reward, is_terminal

    def reset(self):
        self.state = 0


