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
import logging
import sys, subprocess
import numpy as np
import os

import wandb

# os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7"

sys.path.insert(0, "./")
from light_malib.utils.logger import Logger
import ray, random
from ray import serve
import argparse
from light_malib.utils.cfg import load_cfg, convert_to_easydict
from light_malib.utils.random import set_random_seed
from light_malib.framework.pbt_runner import PBTRunner
import time
from vlm.utils import MultiBuffer

# os.environ["RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE"] = "1"
import yaml
from omegaconf import OmegaConf

import pathlib

BASE_DIR = str(pathlib.Path(__file__).resolve().parent.parent)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument('--project_name', type=str, default='OFF_ON_MARL', required=False)
    parser.add_argument('--wandb', type=bool, default=False, help='whether to use the wandb', required=False)
    parser.add_argument('--seed', type=int, default=1256, help='random seed', required=False)
    parser.add_argument('--env', type=str, default='academy_3_vs_1_with_keeper', required=False)
    parser.add_argument('--actor_path', type=str,
                        default='direction_predictor/models_2lstmcell/offline_model/mat/actor_reward_0.2.pth',
                        required=False)
    parser.add_argument('--label', type=str, default='test', required=False)
    args = parser.parse_args()
    return args


def get_local_ip_address():
    import socket

    ip_address = socket.gethostbyname(socket.gethostname())
    return ip_address


def start_cluster():
    try:
        # Connect to existing cluster with explicit port
        cluster_start_info = ray.init(address="auto", runtime_env={"env_vars": {"CUDA_VISIBLE_DEVICES": "1,2,3"}})
    except ConnectionError:
        cluster_start_info = ray.init(resources={})

    Logger.warning(
        "============== Cluster Info ==============\n{}".format(cluster_start_info)
    )
    Logger.warning("* cluster resources:\n{}".format(ray.cluster_resources()))
    Logger.warning(
        "this worker ip: {}".format(ray.get_runtime_context().worker.node_ip_address)
    )
    return cluster_start_info


def _get_free_gpu_id(free_size=0.7):
    all_command = "nvidia-smi -q -d Memory |grep -A4 GPU|grep Used"
    all_result = subprocess.getoutput(all_command)
    # free
    if len(all_result) > 0:
        all_data = [float(item.split(':')[1].strip('MiB').strip(' ')) for item in all_result.split('\n')]
        return np.argmin(all_data)
    else:
        return random.randint(0, 2)


def main():
    # cuda_num = _get_free_gpu_id()
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_num)
    # import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    args = parse_args()
    cfg = load_cfg(args.config)
    cfg['training_manager']['master_port'] = random.randint(1000, 9000)  # Keep as int
    args.env = cfg["rollout_manager"]["worker"]["envs"][0]["scenario_config"]["env_name"]
    for arg in vars(args):
        cfg[arg] = getattr(args, arg)
    set_random_seed(cfg.seed)
    cfg['rollout_manager']["seed"] = cfg.seed
    assert cfg.distributed.nodes.master.ip is not None
    cluster_start_info = start_cluster()

    if cfg.distributed.nodes.master.ip == "auto":
        # ip = get_local_ip_address()
        ip = ray.get_runtime_context().worker.node_ip_address
        cfg.distributed.nodes.master.ip = ip
        Logger.warning("Automatically set master ip to local ip address: {}".format(ip))

    # check cfg
    # check gpu number here
    assert (
            cfg.training_manager.num_trainers <= ray.cluster_resources()["GPU"]
    ), "#trainers({}) should be <= #gpus({})".format(
        cfg.training_manager.num_trainers, ray.cluster_resources()["GPU"]
    )
    # check batch size here
    assert (
            cfg.training_manager.batch_size <= cfg.data_server.table_cfg.capacity
    ), "batch_size({}) should be <= capacity({})".format(
        cfg.training_manager.batch_size, cfg.data_server.table_cfg.capacity
    )
    # check sync_training
    if cfg.framework.sync_training and cfg.framework.get('on_policy', True):
        assert cfg.data_server.table_cfg.sample_max_usage == 1
        assert cfg.training_manager.batch_size == cfg.rollout_manager.batch_size
        assert cfg.rollout_manager.worker.sample_length <= 0

    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    cfg.expr_log_dir = os.path.join(
        cfg.log_dir, cfg.expr_group, cfg.expr_name, cfg.label, str(cfg.seed)
    )
    cfg.expr_log_dir = os.path.join(BASE_DIR, cfg.expr_log_dir)
    os.makedirs(cfg.expr_log_dir, exist_ok=True)

    # copy config file
    yaml_path = os.path.join(cfg.expr_log_dir, "config.yaml")
    with open(yaml_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
        # yaml.dump(OmegaConf.to_yaml(cfg), f, sort_keys=False)

    cfg = convert_to_easydict(cfg)

    # add files to wandb artifact
    if cfg.wandb:
        artifact = wandb.Artifact(cfg.label, type='config')
        artifact.add_file(yaml_path)

        artifact.save()

    from light_malib.monitor.monitor import Monitor
    from light_malib.utils.distributed import get_resources

    Monitor = ray.remote(**get_resources(cfg.monitor.distributed.resources))(Monitor)
    monitor = Monitor.options(name="Monitor", max_concurrency=5).remote(cfg)

    Recorder = ray.remote(num_cpus=1)(MultiBuffer)
    recorder = Recorder.options(name="Recorder", max_concurrency=100).remote()

    runner = PBTRunner(cfg)

    try:
        runner.run()
    except KeyboardInterrupt as e:
        Logger.warning(
            "Detected KeyboardInterrupt event, start background resources recycling threads ..."
        )
    finally:
        runner.close()
        ray.get(monitor.close.remote())
        ray.shutdown()


if __name__ == "__main__":
    main()
