from typing import Sequence, Callable, Tuple, Optional

import torch
from torch import nn
import random
import numpy as np

import cs285.infrastructure.pytorch_util as ptu


class DQNAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        num_actions: int,
        make_critic: Callable[[Tuple[int, ...], int], nn.Module],
        make_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        make_lr_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        discount: float,
        target_update_period: int,
        use_double_q: bool = False,
        clip_grad_norm: Optional[float] = None,
    ):
        super().__init__()

        self.critic = make_critic(observation_shape, num_actions)
        self.target_critic = make_critic(observation_shape, num_actions)
        self.critic_optimizer = make_optimizer(self.critic.parameters())
        self.lr_scheduler = make_lr_schedule(self.critic_optimizer)

        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.discount = discount
        self.target_update_period = target_update_period
        self.clip_grad_norm = clip_grad_norm
        self.use_double_q = use_double_q

        self.critic_loss = nn.MSELoss()

        self.update_target_critic()

    def get_action(self, observation: np.ndarray, epsilon: float = 0.02) -> int:
        """
        Used for evaluation.
        """
        observation = ptu.from_numpy(np.asarray(observation))[None]

        # TODO(student): get the action from the critic using an epsilon-greedy strategy
        if random.random() < epsilon:
            # choose random action
            action = torch.tensor([[np.random.randint(self.num_actions)]])
        else:
            action = self.critic(observation).max(1).indices.view(1, 1)

        return ptu.to_numpy(action).squeeze(0).item()

    def update_critic(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict:
        """Update the DQN critic, and return stats for logging."""
        (batch_size,) = rewards.shape

        # Compute target values
        with torch.no_grad():
            # TODO(student): compute target values
            # y_i = r(s_i,a_i) + \gamma \max_{a_i^{'}}Q_{\phi^{'}}(s_i^{'}, a_i^{'})

            #next_qa_values = self.target_critic(next_observations).max(1).values
            next_q_values = self.target_critic(next_observations).max(1).values

            if self.use_double_q:
                raise NotImplementedError
            else:
                next_action = self.critic(next_observations).max(1).indices
            
            #next_q_values = next_qa_values.max(1).values
            target_values = rewards + self.discount * next_q_values

        # TODO(student): train the critic with the target values
        # y_i = r(s_i,a_i) + \gamma \max_{a_i^{'}}Q_{\phi^{'}}(s_i^{'}, a_i^{'})

            
        qa_values = self.critic(observations)
        # Compute from the data actions; see torch.gather
        print("shape: ", qa_values.shape, actions.shape)
        q_values = qa_values.gather(1, actions)
        loss = self.critic_loss(q_values, target_values)


        self.critic_optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
            self.critic.parameters(), self.clip_grad_norm or float("inf")
        )
        self.critic_optimizer.step()

        self.lr_scheduler.step()

        return {
            "critic_loss": loss.item(),
            "q_values": q_values.mean().item(),
            "target_values": target_values.mean().item(),
            "grad_norm": grad_norm.item(),
        }

    def update_target_critic(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    # def update(
    #     self,
    #     obs: torch.Tensor,
    #     action: torch.Tensor,
    #     reward: torch.Tensor,
    #     next_obs: torch.Tensor,
    #     done: torch.Tensor,
    #     step: int,
    # ) -> dict:
    #     """
    #     Update the DQN agent, including both the critic and target.
    #     """
    #     # TODO(student): update the critic, and the target if needed

    #     return critic_stats
