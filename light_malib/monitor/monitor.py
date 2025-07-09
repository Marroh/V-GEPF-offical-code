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

from light_malib.utils.logger import Logger
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wandb

class Monitor:
    """
    TODO(jh): wandb etc
    TODO(jh): more functionality.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.monitor_type = cfg.monitor.get('type', 'local')
        # if self.monitor_type == 'local':
        self.writer = SummaryWriter(log_dir=cfg.expr_log_dir)
        if cfg.wandb:
            wandb_name = f'seed_{cfg.seed}_{cfg.label}'
            wandb.login(key="xxx", relogin=True)
            wandb_run=wandb.init(entity='LLMGuidedSoccer', project=cfg.project_name, name=wandb_name, group=cfg.env, job_type=cfg.label, config=cfg)
            wandb_run.log_code("./",
                               include_fn=lambda path: path.endswith("train_main_goal.py") or path.endswith(".ipynb"))

    def get_expr_log_dir(self):
        return self.cfg.expr_log_dir

    def add_scalar(self, tag, scalar_value, global_step, *args, **kwargs):
        if self.monitor_type == 'local':
            self.writer.add_scalar(tag, scalar_value, global_step, *args, **kwargs)
        if self.cfg.wandb:
            wandb.log({tag: scalar_value, 'global step1': global_step})

    def add_scalar_simple(self, tag, scalar_value, step, *args, **kwargs):
        if self.monitor_type == 'local':
            self.writer.add_scalar(tag, scalar_value, step, *args, **kwargs)
        if self.cfg.wandb:
            wandb.log({tag: scalar_value, 'global step2': step})

    def add_text_simple(self, tag, skill, response, step, prompt, *args, **kwargs):
        if self.monitor_type == 'local':
            pass
        if self.cfg.wandb:
            if not hasattr(self, 'table_data'):
                print('>> Create table_data')
                self.table_data = []
            self.table_data.append([step, prompt, response, skill])
            table = wandb.Table(columns=["Step", "Prompt", "Response", "Skill"], data=self.table_data)
            wandb.log({tag: table, 'global step2': step})

    def add_text_multiple(self, tag, step, **kwargs):
        if self.monitor_type == 'local':
            pass
        if self.cfg.wandb:
            if not hasattr(self, 'table_data'):
                print('>> Create table_data')
                self.table_data = []
            columns = []
            for key, text in kwargs.items():
                self.table_data.append([text])
                columns.append(key)
            table = wandb.Table(columns=columns, data=self.table_data)
            wandb.log({tag: table, 'global step2': step})

    def add_video(self, tag, step, data):
        """
        Log a video to wandb.
        
        Args:
            tag (str): Name for the video
            step (int): Current step
            data: Can be either:
                - numpy array with shape (time, channel, height, width) or (batch, time, channel, height, width)
                - list of PIL images
                - path to a video file
        """
        if self.monitor_type == 'local':
            pass
        if self.cfg.wandb:
            if isinstance(data, list) and hasattr(data[0], 'convert'):  # Check if it's a list of PIL images
                # Convert PIL images to numpy array with shape (time, height, width, channel)
                frames = np.stack([np.array(img) for img in data])
                # Transpose to (time, channel, height, width) as required by wandb
                frames = np.transpose(frames, (0, 3, 1, 2))
                wandb.log({tag: wandb.Video(frames, fps=4), 'global step2': step})
            elif isinstance(data, np.ndarray):
                # For numpy arrays, specify fps
                wandb.log({tag: wandb.Video(data, fps=4), 'global step2': step})
            else:
                # For file paths
                wandb.log({tag: wandb.Video(data), 'global step2': step})

    def add_multiple_scalars(
        self, main_tag, tag_scalar_dict, global_step, *args, **kwargs
    ):
        for tag, scalar_value in tag_scalar_dict.items():
            tag = main_tag + tag
            if self.monitor_type == 'local':
                self.writer.add_scalar(tag, scalar_value, global_step, *args, **kwargs)
            if self.cfg.wandb:
                wandb.log({tag: scalar_value})
                if "win" in tag:
                    wandb.log({tag: scalar_value, 'global step': global_step})

    def add_scalars(self, main_tag, tag_scalar_dict, global_step, *args, **kwargs):
        if self.monitor_type == 'local':
            self.writer.add_scalars(main_tag, tag_scalar_dict, global_step, *args, **kwargs)
        if self.cfg.wandb:

            log_dict = {}
            for tag, scalar in tag_scalar_dict.items():
                log_dict[f'{main_tag}_{tag}'] = scalar
            wandb.log(log_dict)

        wandb.log({ 'global step3': global_step})

    def add_array(
        self, main_tag, image_array, xpid, ypid, global_step, color, *args, **kwargs
    ):
        array_to_rgb(
            self.writer, main_tag, image_array, xpid, ypid, global_step, color, self.monitor_type,**kwargs
        )

    def close(self):
        self.writer.close()
        wandb.finish()


def array_to_rgb(writer, tag, array, xpid, ypid, steps, color="bwr", mode='local',**kwargs):
    matrix = np.array(array)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    color_map = plt.cm.get_cmap(color)
    cax = ax.matshow(matrix, cmap=color_map)
    ax.set_xticklabels([""] + xpid, rotation=90)
    ax.set_yticklabels([""] + ypid)

    if kwargs.get("show_text", False):
        for (j, i), label in np.ndenumerate(array):
            ax.text(i, j, f"{label:.1f}", ha="center", va="center")

    fig.colorbar(cax)
    ax.grid(False)
    plt.tight_layout()

    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = img / 255.0
    img = img.transpose(2, 0, 1)

    if mode == 'local':
        writer.add_image(tag, img, steps)
    elif mode == 'remote':
        wandb.log({tag: plt})
    else:
        Logger.warning("monitor mode is not implemented")
        pass

    plt.close(fig)
