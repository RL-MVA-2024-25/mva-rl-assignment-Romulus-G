from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from fast_env_py import FastHIVPatient

from evaluate import evaluate_HIV, evaluate_HIV_population
from replay_buffer import ReplayBuffer
from copy import deepcopy
import os
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn

# env = TimeLimit(env=HIVPatient(domain_randomization=False), max_episode_steps=200)
env = TimeLimit(env=FastHIVPatient(domain_randomization=False), max_episode_steps=200)

class ProjectAgent:
    def act(self, state, use_random=False):
        with torch.no_grad():
            return self.model(torch.Tensor(state).unsqueeze(0)).argmax().item()

    def save(self, filename):
        path = os.path.join("model_weights", filename)
        torch.save(self.model.state_dict(), path)

    def load(self):
        path = os.path.join("model_weights", "weights.pt")
        self.model = self.DQN()
        self.model.load_state_dict(torch.load(path, map_location='cpu'))
        self.model.eval()

    def DQN(self):
        dim_state = env.observation_space.shape[0]
        n_actions = env.action_space.n
        return torch.nn.Sequential(
            nn.Linear(dim_state, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),            
            nn.Linear(256, 256),
            nn.ReLU(),            
            nn.Linear(256, 256),
            nn.ReLU(),            
            nn.Linear(256, 256),
            nn.ReLU(),            
            nn.Linear(256, n_actions)
            )

    def train(self):
        
        # Training parameters.
        gamma = 0.98
        batch_size = 700
        learning_rate = 0.001
        buffer_size = 100_000
        eps_min = 0.02
        eps_max = 1.
        eps_decay_period = 20_000
        eps_delay_decay = 100
        update_target_freq = 400

        eps_step = (eps_max - eps_min) / eps_decay_period
        
        self.n_actions = env.action_space.n
        self.memory = ReplayBuffer(buffer_size)
        self.model = self.DQN()
        self.target_model = deepcopy(self.model)
        self.loss_function = torch.nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)    

        # Training loop.
        max_episode = 200
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        prev_validation_score = 0
        state, _ = env.reset()
        eps = eps_max
        step = 0
        while episode < max_episode:
            if step > eps_delay_decay:
                eps = max(eps_min, eps - eps_step)
            
            if np.random.rand() < eps:
                action = env.action_space.sample()
            else:
                action = self.act(state)
            
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            
            for _ in range(3):
                if len(self.memory) > batch_size:
                    X, A, R, X_, D = self.memory.sample(batch_size)
                    QX_max = self.target_model(X_).max(1)[0].detach()
                    update = torch.addcmul(R, 1-D, QX_max, value=gamma)
                    QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
                    loss = self.loss_function(QXA, update.unsqueeze(1))
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            if step % update_target_freq == 0: 
                self.target_model.load_state_dict(self.model.state_dict())
            
            episode_cum_reward += reward
            step += 1

            if not done and not trunc:
                state = next_state
            else:
                episode += 1
                state, _ = env.reset()

                validation_score = evaluate_HIV(agent=self, nb_episode=1)
                if validation_score > prev_validation_score:
                    prev_validation_score = validation_score
                    self.best_model = deepcopy(self.model)
                    self.save("intermediate_weights.pt")

                print(f"Ep {episode} | Eps {eps:.3f} | Buffer Size {len(self.memory)} | Ep. Cum. Rew. {episode_cum_reward:.2e} | Val. Score {validation_score:.2e}")

                episode_return.append(episode_cum_reward)                
                episode_cum_reward = 0                
        # end while

        self.model.load_state_dict(self.best_model.state_dict())
        self.save("weights.pt")
        return episode_return
    
def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    seed_everything(seed=42)
    agent = ProjectAgent()
    agent.train()