# Imports:
# --------
import torch
import random
from collections import deque
import torch.nn.functional as F


# Repla Buffer:
# -------------
class ReplayBuffer():
    def __init__(self, buffer_limit):
        self.buffer = deque(maxlen=buffer_limit)
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)
    
    def size(self):
        return len(self.buffer)


# Train function:
# ---------------
def train(q_net, 
          q_target, 
          memory, 
          optimizer,
          batch_size,
          gamma):
    
    #! We sample from the same Replay Buffer n=10 times
    for _ in range(10):
        #! Monte Carlo sampling of a batch
        s, a, r, s_prime, done_mask = memory.sample(batch_size)

        #! Get the Q-values
        q_out = q_net(s)

        #! DQN update rule
        q_a = q_out.gather(1, a)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
