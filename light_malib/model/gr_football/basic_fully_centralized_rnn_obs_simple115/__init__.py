# Copyright 2022 Digital Brain Laboratory, Yan Song and He jiang
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from light_malib.algorithm.common import actor
from light_malib.algorithm.common import critic
from light_malib.envs.gr_football.encoders.encoder_simple115 import FeatureEncoder
import torch
import torch.nn as nn

class Backbone(nn.Module):
    def __init__(self, model_config, observation_space, action_space, custom_config, initialization):
        super().__init__()
        self.feat_dim = observation_space.shape[0]
        self.hidden_size = model_config.get("embed_dim", 64)
        self.num_players = custom_config["num_agents"]
        
        # Shared GRUCell layer instead of GRU
        self.rnn = nn.GRUCell(self.feat_dim, self.hidden_size)
        
    def forward(self, state, obs, rnn_states, masks):
        # Add input validation
        if obs is None or rnn_states is None:
            raise ValueError("observations and rnn_states cannot be None")
            
        batch_size = obs.shape[0] // self.num_players
        obs = torch.as_tensor(obs, dtype=torch.float32)
        rnn_states = torch.as_tensor(rnn_states, dtype=torch.float32)
        
        # Ensure shapes are correct before reshaping
        expected_rnn_shape = (-1, self.hidden_size)
        if rnn_states.shape[-1] != self.hidden_size:
            raise ValueError(f"Expected rnn_states with last dimension {self.hidden_size}, got {rnn_states.shape[-1]}")
            
        rnn_states = rnn_states.reshape(-1, self.hidden_size)
        rnn_states = self.rnn(obs, rnn_states)
        
        # feats = torch.cat([rnn_states, obs], dim=-1)
        
        return rnn_states
        

class Critic(critic.Critic):
    def __init__(self, model_config, observation_space, action_space, custom_config, initialization, backbone):
        super().__init__(model_config, observation_space, action_space, custom_config, initialization)
        
        self.backbone = backbone
        self.out = nn.Linear(self.backbone.hidden_size, self.act_dim)
        self.num_players = custom_config["num_agents"]
        
    def forward(self, obs, rnn_states, masks):
        # Use shared backbone
        feats = self.backbone(None, obs, rnn_states, masks)
        # print('criticfeats.shape', feats.shape)
        
        # Generate values
        values = self.out(feats)
        
        return values, feats.reshape(self.num_players, -1, self.backbone.hidden_size)

class Actor(actor.Actor):
    def __init__(self, model_config, observation_space, action_space, custom_config, initialization, backbone):
        super().__init__(model_config, observation_space, action_space, custom_config, initialization)
        
        self.backbone = backbone
        self.out = nn.Linear(self.backbone.hidden_size, self.act_dim)
        
    def forward(self, obs, actor_rnn_states, rnn_masks, action_masks, explore, actions):
        # Use shared backbone
        feats = self.backbone(None, obs, actor_rnn_states, rnn_masks)

        # print('actor feats.shape', feats.shape)
        # print('actor actor_rnn_states.shape', actor_rnn_states.shape)
        
        # Generate action logits
        logits = self.out(feats)
        
        # Apply action masking
        illegal_action_mask = 1 - action_masks
        logits = logits - 1e10 * illegal_action_mask

        # Create distribution and sample actions
        dist = torch.distributions.Categorical(logits=logits)
        if actions is None:
            actions = dist.sample() if explore else dist.probs.argmax(dim=-1)
            dist_entropy = None
        else:
            dist_entropy = dist.entropy()
        action_log_probs = dist.log_prob(actions)
        
        return actions, actor_rnn_states, action_log_probs, dist_entropy 