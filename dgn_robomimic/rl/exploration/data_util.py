import numpy as np
import torch


class Buffer:

    def __init__(self):

        #Main storage
        self.states = []
        self.actions = []

        #Episode storage
        self.ep_states = []
        self.ep_actions = []


    def add_transition(self, state, action):

        self.ep_states.append(state)
        self.ep_actions.append(action)

    def end_episode(self, success):

        if success:
            self._store_episode()

        #Reset Epiosode
        self.ep_states = []
        self.ep_actions = []

    def _store_episode(self):
        
        #Stack
        ep_states = torch.stack(self.ep_states, axis=0).squeeze(1).cpu()
        ep_actions = torch.stack(self.ep_actions, axis=0).cpu()


        self.states.append(ep_states)
        self.actions.append(ep_actions)

    
    @property
    def num_eps(self):
        return len(self.states)


    @property
    def has_data(self):
        return self.num_eps > 0
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        
        entries = []
        for i in range(self.states[idx].shape[0]):
            entry = {'state': self.states[idx][i], 'action': self.actions[idx][i]}
            entries.append(entry)

        return entries
        

        


    