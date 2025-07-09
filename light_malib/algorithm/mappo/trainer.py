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
import itertools
import random

import numpy as np
from collections import defaultdict

import ray

from light_malib.training.data_generator import (
    recurrent_generator,
    simple_data_generator,
)
from .loss import MAPPOLoss
import torch
import functools
from light_malib.utils.logger import Logger
from light_malib.utils.timer import global_timer
from ..return_compute import compute_return
from ..common.trainer import Trainer
from light_malib.registry import registry
from light_malib.utils.episode import EpisodeKey
from ...vlm.utils import RealTimeDrawer, vLLMAgent
from ...vlm.vlm_critic import CLIPCritic


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


@registry.registered(registry.TRAINER)
class MAPPOTrainer(Trainer):
    def __init__(self, tid):
        super().__init__(tid)
        self.v11 = None
        self.id = tid
        # TODO(jh)
        self._loss = MAPPOLoss()
        self.change_period = 20
        self.need_normalization = True
        self.drawer = RealTimeDrawer()
        self.vllm = vLLMAgent(model='localminiCPM', memory_size=self.change_period)
        self.rollout_manger = ray.get_actor("RolloutManager")
        self.monitor = ray.get_actor("Monitor")
        self.recorder = ray.get_actor("Recorder")


        # self.skill_coef = {
        #         'has advantage': 0.5,
        #         'cover wider pitch': 0.5,
        #         'correct formation': 0.5,
        #         'encourage possessing': 0.5,
        #         'encourage dribbling': 0.5,
        #         'encourage attack': 0.5,
        #         'encourage defense': 0.5,
        #         # 'encourage cooperative passing': 2.0,
        # }

        self.skill_coef = {
                'has advantage': 0.1,
                # 'cover wider pitch': 0.1,
                'correct formation': 0.1,
                # 'encourage possessing': 0.1,
                'encourage dribbling': 0.1,
                'encourage attack': 0.1,
                'encourage defense': 0.1
                # 'encourage passing': 2.0,
        }

    def optimize(self, batch, **kwargs):
        # ray.util.pdb.set_trace()
        # Note: preprocess for VLM-MAPPO
        batch = self.preprocess(batch, **kwargs)

        total_opt_result = defaultdict(lambda: 0)
        policy = self.loss.policy

        ppo_epoch = policy.custom_config["ppo_epoch"]
        num_mini_batch = policy.custom_config["num_mini_batch"]  # num_mini_batch
        kl_early_stop = policy.custom_config.get("kl_early_stop", None)
        assert (
                kl_early_stop is None
        ), "TODO(jh): kl early stop is not supported is current distributed implementation."

        # move data to gpu
        global_timer.record("move_to_gpu_start")
        for key, value in batch.items():
            if isinstance(value, np.ndarray):
                value = torch.FloatTensor(value)
            batch[key] = value.to(policy.device)

        if EpisodeKey.CUR_STATE not in batch:
            batch[EpisodeKey.CUR_STATE] = batch[EpisodeKey.CUR_OBS]
        global_timer.time("move_to_gpu_start", "move_to_gpu_end", "move_to_gpu")

        kl_diff = 0
        for i_epoch in range(ppo_epoch):
            # NOTE(jh): for backward compatibility, when return_mode="new_gae", only call return_compute once.
            if i_epoch == 0 or policy.custom_config["return_mode"] in ["new_gae_trace"]:
                batch_with_return = self._compute_return(policy, batch)

            data_generator_fn = self._get_data_generator(policy, batch_with_return, num_mini_batch)

            for mini_batch in data_generator_fn():
                global_timer.record("loss_start")
                tmp_opt_result = self.loss(mini_batch)
                global_timer.time("loss_start", "loss_end", "loss")
                for k, v in tmp_opt_result.items():
                    total_opt_result[k] = v

            if i_epoch == 0:
                start_kl = tmp_opt_result["approx_kl"]
            else:
                kl_diff += tmp_opt_result["approx_kl"] - start_kl
                start_kl = tmp_opt_result["approx_kl"]

            if (
                    kl_early_stop is not None
                    and tmp_opt_result["approx_kl"] > kl_early_stop
            ):
                break

            total_opt_result["kl_diff"] = kl_diff
            total_opt_result["training_epoch"] = i_epoch + 1

        return total_opt_result

    def preprocess(self, batch, **kwargs):
        # TODO [Done]: create global epoch count to trigger `Skill Advising`
        print(f'>> Training Epoch {ray.get(self.rollout_manger.get_rollout_epoch.remote())}')
        if ray.get(self.rollout_manger.get_rollout_epoch.remote()) % self.change_period == 0:
            # Note: add pass info
            total_pass = ray.get(self.recorder.get.remote('num_pass'))
            good_pass = ray.get(self.recorder.get.remote('good_pass'))
            total_shot = ray.get(self.recorder.get.remote('total_shot'))
            win = ray.get(self.recorder.get.remote('win'))
            self.vllm.memory.record_statistic({'total shot': total_shot,
                                               'win': win
                                               })
            images = ray.get(self.recorder.get.remote('images'))
            ray.get(self.recorder.remove.remote('images'))
            self.vllm.memory.record_imgs(images)
            self.vllm.choose_skill()

            # log skill
            ray.get(
                self.monitor.add_text_simple.remote(
                    'potential_reward/conversation',
                    self.vllm.memory.last_skill,
                    self.vllm.response,
                    self.rollout_manger.get_rollout_epoch.remote(),
                    self.vllm.log_prompt
                )
            )

            ray.get(
                self.monitor.add_scalar_simple.remote(
                    'potential_reward/choose_checkpoint',
                    np.random.randint(0, 100),
                    self.rollout_manger.get_rollout_epoch.remote()
                )
            )
            print(f'>> Skill Advising: {self.vllm.memory.last_skill}\n{self.vllm.response}')

            self.need_normalization = True

        actions = batch[EpisodeKey.ACTION]
        observation = batch[EpisodeKey.GLOBAL_STATE]  # [32, 3002, 1, 128]
        workers, episode_len, _, feature_len = observation.shape

        # extract infos for drawing
        # TODO: Modify to adapt to 5v5
        if feature_len == 128:
            print('>> 11 v 11')
            self.v11 = True
        elif feature_len == 68:
            print('>> 5 v 5')
            self.v11 = False
            
        last_actions = np.roll(actions, 1, axis=1)  # [32, 3002, 10]
        last_actions[:, 0] = 0  # set the last action at t0 to 0 (idle)

        if self.v11:
            last_actions = last_actions.reshape(-1, 10)
            left_pos = observation[..., :22].reshape(-1, 11, 2)
            left_role = observation[..., 44:55].reshape(-1, 11)
            right_pos = observation[..., 55:77].reshape(-1, 11, 2)
            right_role = observation[..., 99:110].reshape(-1, 11)
            ball_pos = observation[..., 110:113].reshape(-1, 3)
        else:
            last_actions = last_actions.reshape(-1, 4)
            left_pos = observation[..., :10].reshape(-1, 5, 2)
            left_role = observation[..., 20:25].reshape(-1, 5)
            right_pos = observation[..., 25:35].reshape(-1, 5, 2)
            right_role = observation[..., 45:50].reshape(-1, 5)
            ball_pos = observation[..., 50:53].reshape(-1, 3)

        assert left_pos.shape[0] == right_pos.shape[0] == ball_pos.shape[0]

        # TODO [Done]: need a potential-based reward in shape [32, 3002, 10, 1]
        poten_rews = self._compute_potential_reward(left_pos, left_role, right_pos, right_role, ball_pos, workers,
                                                    last_actions, episode_len)

        # passing_rews = self._compute_passing_reward(last_action=batch[EpisodeKey.ACTION])

        batch[EpisodeKey.REWARD] = batch[EpisodeKey.REWARD] + poten_rews

        return batch

    def _compute_potential_reward(self, left_pos, left_role, right_pos, right_role, ball_pos, workers, last_actions, episode_len):
        # TODO [Done]: make batch: reshape [32, 3002, 1, 128] to [32 * 3002, 128]
        # TODO [Done]: draw batch images
        obs_dataset = []
        # print('Debug', self.vllm.memory.skill_repo)
        for idx in range(left_pos.shape[0]):
            formated_obs = {
                'left_team': left_pos[idx],
                'left_team_roles': left_role[idx],
                'right_team': right_pos[idx],
                'right_team_roles': right_role[idx],
                'ball': ball_pos[idx],
                'last_actions': last_actions[idx],  # [3002, 1]
                'skill': self.vllm.memory.skill_repo[self.vllm.memory.last_skill],
            }
            # print('skill:', formated_obs['skill'])
            obs_dataset.append(formated_obs)

        ds = ray.data.from_items(obs_dataset)
        # print(f'>> {ds.schema()} {ds.count()}')
        preprocess_ds = ds.map(
            RealTimeDrawer,
            num_cpus=1,
            concurrency=32,
        )  # 20s for 10w images

        # print(f'>> {preprocess_ds.schema()} {ds.count()}')

        # Note: rewards shape [32 * 3002, 1]
        rewards_ds = preprocess_ds.map_batches(
            CLIPCritic,
            num_gpus=0.3,
            batch_size=1024,
            concurrency=8,
        )

        rewards = ray.get(rewards_ds.to_numpy_refs())

        rewards_arr = []
        for d in rewards:
            rewards_arr.extend(d['rewards'])

        rewards_arr = np.array(rewards_arr)
        # TODO [Done]: try magnifying it
        # TODO [Done]: Need normalization for each skill's potential function
        if self.need_normalization:
            self.rewards_mean = np.mean(rewards_arr)
            self.rewards_std = np.std(rewards_arr)
            self.need_normalization = False

        coef = self.skill_coef[self.vllm.memory.last_skill]
        rewards_arr = ((rewards_arr - self.rewards_mean) / self.rewards_std) * coef

        # compute potential-based reward
        shift_rewards = np.zeros_like(rewards_arr)
        shift_rewards[:-1] = rewards_arr[1:]
        potential_rewards = 0.995 * shift_rewards - rewards_arr

        potential_rewards = potential_rewards.reshape(workers, episode_len, -1)
        if self.v11:
            potential_rewards = np.repeat(potential_rewards, 10, axis=2)[..., np.newaxis]
        else:
            # 5v5
            potential_rewards = np.repeat(potential_rewards, 4, axis=2)[..., np.newaxis]


        # TODO [Done]: record info for VLM memory
        # record & log
        mean_poten_rews = np.round(np.mean(rewards_arr), 2)  # scale to avoid all zero
        ray.get(
            self.monitor.add_scalar_simple.remote(
                f'potential_reward/mean_{self.vllm.memory.last_skill}',
                mean_poten_rews,
                self.rollout_manger.get_rollout_epoch.remote()
            )
        )
        self.vllm.memory.record_reward(mean_poten_rews)

        # TODO[Done]: CLIP batch inference

        return potential_rewards

    def _compute_return(self, policy, batch):
        # compute return
        global_timer.record("compute_return_start")
        new_batch = compute_return(policy, batch)
        global_timer.time(
            "compute_return_start", "compute_return_end", "compute_return"
        )
        return new_batch

    def _get_data_generator(self, policy, new_batch, num_mini_batch):
        # build data generator
        if policy.custom_config["use_rnn"]:
            data_generator_fn = functools.partial(
                recurrent_generator,
                new_batch,
                num_mini_batch,
                policy.custom_config["rnn_data_chunk_length"],
                policy.device,
            )
        else:
            data_generator_fn = functools.partial(
                simple_data_generator, new_batch, num_mini_batch, policy.device
            )

        return data_generator_fn
