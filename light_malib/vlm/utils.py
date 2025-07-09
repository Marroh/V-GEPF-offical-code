import base64
import logging
import os.path
import re
import time
from io import BytesIO
os.environ["SDL_AUDIODRIVER"] = "dummy"

import ray
import torch
import requests
from fastapi import FastAPI
from ray import serve
from requests import Request
from sklearn.cluster import KMeans
from mplsoccer.pitch import Pitch
import matplotlib.pyplot as plt
import numpy as np
import PIL
import openai

# from light_malib.vlm.vlm_critic import LocalMiniCPM

os.environ['OPENAI_API_KEY'] = 'xxx'
os.environ["OPENAI_BASE_URL"] = 'xxx'


class AnyBuffer:
    def __init__(self, max_size=16):
        self.max_size = max_size
        self.data = []

    def is_full(self):
        return len(self.data) >= self.max_size

    def add(self, item):
        if len(self.data) >= self.max_size:
            self.data.pop(0)
        self.data.append(item)

    def clear(self):
        self.data = []


class MultiBuffer:
    def __init__(self):
        self.buffer = {}

    def add(self, data, max_len=1):
        # print('>> request keys: ', request.query_params.keys())
        # ray.util.pdb.set_trace()
        for k, v in data.items():
            if k not in self.buffer:
                self.buffer[k] = AnyBuffer(max_len)
            self.buffer[k].add(v)

    def get(self, key):
        if key not in self.buffer:
            return None
        return self.buffer[key].data

    def remove(self, key):
        if key in self.buffer:
            self.buffer.pop(key)


class vLLMMemory:
    def __init__(self, max_size=50):
        self.pb_rewards = AnyBuffer(max_size)
        self.img_buffer = AnyBuffer(3000)
        self.img = None
        self.last_skill = 'encourage attack'
        self.stats_prompt = ''
        # skill name and detailed description
        self.skill_repo = {
            'has advantage': 'The blue team has bigger advantage than the red team',
            # 'cover wider pitch': 'blue area covers a wider pitch than the red one',
            'correct formation': 'three blue formation lines are parallel and have proper spacing',
            # 'encourage possessing': 'the black ball is possessed by the blue team',
            'encourage dribbling': 'the black ball is well dribbled by the blue team',
            # 'encourage attack': 'the black ball is close to the red team goal (on the right side)',
            'encourage attack': 'the blue team is performing a coordinated attack',
            # 'encourage passing': 'the blue team is passing',
            'encourage defense': 'the blue team is trying to defend when the ball is close to their goal',
        }
        self.skill_names = ','.join(self.skill_repo.keys())
        self.skill_description = '\n'.join([f'{k}: {v}' for k, v in self.skill_repo.items()])

    def record_reward(self, reward):
        self.pb_rewards.add(reward)

    def record_statistic(self, statistic):
        # TODO: record `total_pass`
        self.stats = statistic
        # round all
        for k, v in self.stats.items():
            self.stats[k] = [x for x in v]
        self.stats_prompt = '\n'.join([f'{k}: [{", ".join(v)}]' for k, v in self.stats.items()])

    def record_img(self, img):
        self.img_buffer.add(img)

    def record_imgs(self, imgs):
        for img in imgs:
            self.img_buffer.add(img)

    def dump_to_video(self, img):
        # turn a list of image in video
        pass

    def get(self):
        # turn list into string
        rewards = ', '.join([str(reward) for reward in self.pb_rewards.data])
        prompt = ('Analyse this video.\n'
                  'We want to guide the blue team to master human-like football skills. Here are some useful information:\n'
                  'Here are some records:\n{2}\n'
                  'Please carefully analyze the record and current video, then provide advice for the next skill.\n'
                  'You should choose one from [{3}]. Here are their descriptions:\n'
                  '{4}\n'
                  'The last skill is {0}. Its reward record is [{1}]. What is the next skill to master? Try not repeating the last skill.\n'
                  'Output Format:\n'
                  '\'\'\'\nAnalysis: ...\nNext Skill: ...\n\'\'\'').format(self.last_skill, rewards, self.stats_prompt, self.skill_names,
                                                      self.skill_description)
        return prompt, self.img_buffer.data[0]


class vLLMAgent:
    def __init__(self, model='gpt-4o', memory_size=50):
        self.log_prompt = None
        self.response = None
        self.model = model
        self.memory = vLLMMemory(memory_size)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.info('vLLM agent initialized')

        # pre-load
        if model == 'localminiCPM':
            self.logger.info('Local miniCPM model is used')
            self.vllm = LocalMiniCPM(device='auto')

    def get_response(self):
        print(f'>> You are using {self.model}')
        if self.model == 'gpt-4o':
            prompt, img = self.memory.get()

            self.log_prompt = prompt

            buffer = BytesIO()
            img.save(buffer, format="JPEG")
            img_bytes = buffer.getvalue()
            base64_image = base64.b64encode(img_bytes).decode('utf-8')

            vllm = openai.OpenAI()
            content = []

            content.append({"type": "text", "text": prompt})
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    # "detail": "low"  # note: consume 80 tokens per image
                }
            })

            try:
                response_obj = vllm.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "user",
                            "content": content,
                        }
                    ],
                    max_tokens=300,
                )
                # extract skill name
                pattern = '|'.join([re.escape(skill) for skill in self.memory.skill_repo.keys()])
                matched_skills = re.findall(pattern, response_obj.choices[0].message.content.lower())
                # print(pattern)
                # print(response_obj.choices[0].message.content)
                # print(f'Skill: {matched_skills}')
                if matched_skills != []:
                    self.memory.last_skill = matched_skills[0]
                    return True, matched_skills[0]
                else:
                    return False, ''

            except Exception as e:
                self.logger.error(f'Error in get_response: {e}')
                return False, ''

        # miniCPM run in backend via ray [discarded]
        elif self.model == 'miniCPM':
            prompt, img = self.memory.get()

            self.log_prompt = prompt

            buffer = BytesIO()
            img.save(buffer, format="JPEG")
            img_bytes = buffer.getvalue()
            base64_image = base64.b64encode(img_bytes).decode('utf-8')

            try:
                self.response = requests.get("http://localhost:8000/chat",
                                        params={"img_base64": base64_image, "question": prompt}).json()

                pattern = '|'.join([re.escape(skill) for skill in self.memory.skill_repo.keys()])
                matched_skills = re.findall(pattern, self.response)
                # print(pattern)
                # print(response)
                # print(f'Skill: {matched_skills}')
                if matched_skills != []:
                    return True, matched_skills[-1]
                else:
                    return False, ''

            except Exception as e:
                self.logger.error(f'{e}. Response:{self.response}')
                return False, ''

        elif self.model == 'localminiCPM':
            prompt, img = self.memory.get()

            self.log_prompt = prompt

            images_list = self.memory.img_buffer.data[0]
            try:
                self.response = self.vllm.chat(images_list, prompt)

                sub_pattern = '|'.join([re.escape(skill) for skill in self.memory.skill_repo.keys()])
                pattern = r"next skill:\s*({})".format(sub_pattern)
                matched_skills = re.findall(pattern, self.response.lower())
                # print(pattern)
                print(self.response)
                print(f'Skill: {matched_skills}')
                if matched_skills != []:
                    return True, matched_skills[-1]
                else:
                    return False, ''

            except Exception as e:
                self.logger.error(f'{e}. Response:{self.response}')
                return False, ''

    def choose_skill(self, random=False):
        if random:
            skill = np.random.choice(list(self.memory.skill_repo.keys()))
            self.log_prompt = 'Random'
            self.memory.last_skill = skill
        else:
            succ = False
            while not succ:
                succ, skill = self.get_response()
            self.memory.last_skill = skill

        return self.memory.skill_repo[skill]
    
class SampleRayDataset:
    def __init__(self):
        self.sample_prob = 0.1

    def __call__(self, batch):
        print(f'>> batch: {batch["images"].shape[0]}')
        if np.random.rand() < self.sample_prob:
            return {'images': batch['images'], 'chosen': np.ones(len(batch['images']))}
        else:
            return {'images': batch['images'], 'chosen': np.zeros(len(batch['images']))}


role_list = ['GK', 'CB', 'LB', 'RB', 'DM', 'CM', 'LM', 'RM', 'AM', 'CF']


class RealTimeDrawer:
    def __init__(self):
        self.ax = None
        self.fig = None
        self.pitch = None
        self.left_player = None
        self.right_player = None
        self.ball = None
        self.carrier = None
        self.hull = None
        self.lineups = None
        self.left_line0 = None
        self.left_line1 = None
        self.left_line2 = None
        self.right_line0 = None
        self.right_line1 = None
        self.right_line2 = None
        self.left_patches = []
        self.right_patches = []

        self.pitch_set()
        self.color = ['black', '#DC2B14', 'green', 'red']

    @staticmethod
    def transform(agent_loc, drop_GK=False, team='left'):
        agent_loc = np.array(agent_loc)
        agent_loc = (agent_loc + np.array([1, 0.42])) / np.array([2, 0.84]) * np.array([120, 80])
        if drop_GK:
            idx_GK = agent_loc.argmin() // 2 if team == 'left' else agent_loc.argmax() // 2
            agent_loc = np.delete(agent_loc, idx_GK, axis=0)
        return agent_loc

    @staticmethod
    def cluster_pressureLine(X, k=3):
        """
        Cluster players by x_coordinate

        X: x_coordinate list/array-1d
        k: number of clusters(number of formation lines, typically 3)

        return: centers and labels
        """
        X = np.expand_dims(X, axis=0).T
        cluster = KMeans(n_clusters=int(k), random_state=2147)
        cluster.fit(X)
        centers = cluster.cluster_centers_.reshape(1, -1)[0]
        labels = cluster.labels_
        # print(labels)
        return centers, labels

    def pitch_set(self, left_color='royalblue', right_color='red'):
        self.playerNum = 11

        self.pitch = Pitch(pitch_type='statsbomb',
                           pitch_color='None', line_color='#000000', goal_type='box', linewidth=1,
                           pad_bottom=0.2, pad_top=4)
        self.fig, self.ax = self.pitch.draw()

        self.marker_kwargs = {'marker': 'o',
                              'markeredgecolor': 'black',
                              'linestyle': 'None'}

        self.line_kwargs = {'linestyle': '-',
                            'linewidth': 5,
                            'alpha': 0.5, }

        self.left_player, = self.ax.plot([], [], ms=10, **self.marker_kwargs, markerfacecolor=left_color)
        self.right_player, = self.ax.plot([], [], ms=10, **self.marker_kwargs, markerfacecolor=right_color)
        self.ball, = self.ax.plot([], [], ms=6, **self.marker_kwargs, markerfacecolor='black')
        # self.carrier, = self.ax.plot([], [], ms=13, markeredgecolor='y', markeredgewidth=3, marker='o', zorder=0)
        self.left_line0, = self.ax.plot([], [], ms=0, **self.line_kwargs, color=left_color)
        self.left_line1, = self.ax.plot([], [], ms=0, **self.line_kwargs, color=left_color)
        self.left_line2, = self.ax.plot([], [], ms=0, **self.line_kwargs, color=left_color)
        self.right_line0, = self.ax.plot([], [], ms=0, **self.line_kwargs, color=right_color)
        self.right_line1, = self.ax.plot([], [], ms=0, **self.line_kwargs, color=right_color)
        self.right_line2, = self.ax.plot([], [], ms=0, **self.line_kwargs, color=right_color)

        self.left_text = [self.ax.text(0, 0, '',
                                       size=15,
                                       horizontalalignment='center',
                                       verticalalignment='center') for i in range(self.playerNum)]
        self.right_text = [self.ax.text(0, 0, '',
                                        size=15,
                                        horizontalalignment='center',
                                        verticalalignment='center') for i in range(self.playerNum)]

        self.info_text = self.ax.text(60,
                                      40,
                                      '',
                                      size=200,
                                      horizontalalignment='center',
                                      verticalalignment='center',
                                      zorder=0,
                                      alpha=0.2
                                      )

    def lines_group(self, i, group_loc, team='left'):
        """
        Visualize left-to-right line

        group_loc: loc of a group of 'line'
        team: 'left' or 'right'
        """
        sortY_idxs = np.argsort(group_loc, axis=0)
        sortY_idxs = sortY_idxs[:, 1]
        group_loc = group_loc[sortY_idxs]
        x = []
        y = []
        x_end = []
        y_end = []
        # more than 1 player in one line
        if group_loc.shape[0] > 1:
            for connect in range(group_loc.shape[0] - 1):
                x.append(group_loc[connect, 0])
                y.append(group_loc[connect, 1])
                x_end.append(group_loc[connect + 1, 0])
                y_end.append(group_loc[connect + 1, 1])

        if team == 'left':
            if i == 0:
                self.left_line0.set_data(group_loc[:, 0], group_loc[:, 1])
            if i == 1:
                self.left_line1.set_data(group_loc[:, 0], group_loc[:, 1])
            if i == 2:
                self.left_line2.set_data(group_loc[:, 0], group_loc[:, 1])
        if team == 'right':
            if i == 0:
                self.right_line0.set_data(group_loc[:, 0], group_loc[:, 1])
            if i == 1:
                self.right_line1.set_data(group_loc[:, 0], group_loc[:, 1])
            if i == 2:
                self.right_line2.set_data(group_loc[:, 0], group_loc[:, 1])

    def draw_from_obs(self, obs, simple=False):
        '''
        extract all locations of players from Google Research Football simple115_v2 observation
        include coordinates of left team players, coordinates of right team players, ball position
        check details at https://github.com/google-research/football/blob/master/gfootball/doc/observation.md#simple115_v2
        '''

        #############
        # transform #
        #############
        video_flag = False
        v11 = False
        left_team = obs['left_team']
        right_team = obs['right_team']
        ball = obs['ball'][:2]
        if 'last_actions' in obs:
            last_actions = obs['last_actions']
        else:
            last_actions = None  # do rendering in env.step
        if left_team.shape[0] == 11:
            v11 = True

        # print(f'>> left_team: {left_team.shape}')
        # print(f'>> right_team: {right_team.shape}')
        # print(f'>> ball: {ball.shape}')

        left_team_loc = self.transform(left_team)
        right_team_loc = self.transform(right_team)
        ball_loc = self.transform(ball)

        # left_team_loc_dropGK = self.transform(left_team, drop_GK=True, team='left')
        # right_team_loc_dropGK = self.transform(right_team, drop_GK=True, team='right')

        # left_team_roles = obs['left_team_roles']
        # right_team_roles = obs['right_team_roles']

        ###################
        # player and ball #
        ###################
        # set_data to reload
        self.left_player.set_data(left_team_loc[:, 0], left_team_loc[:, 1])
        self.right_player.set_data(right_team_loc[:, 0], right_team_loc[:, 1])
        self.ball.set_data([ball_loc[0]], [ball_loc[1]])
        # self.carrier.set_data(left_team_loc[5, 0], left_team_loc[5, 1])  # debug
        # for j in range(len(left_team_loc)):
        #     self.left_text[j].set_position([left_team_loc[j, 0], left_team_loc[j, 1]])
        #     self.left_text[j].set_text(s=str(role_list[int(left_team_roles[j])]))
        # for j in range(len(right_team_loc)):
        #     self.right_text[j].set_position([right_team_loc[j, 0], right_team_loc[j, 1]])
        #     self.right_text[j].set_text(s=str(role_list[int(right_team_roles[j])]))

        if simple:
            plt.tight_layout()
            self.fig.canvas.draw()
            img = PIL.Image.frombytes('RGB', self.fig.canvas.get_width_height(), self.fig.canvas.tostring_rgb())

            return img

        # ###############
        # # convex hull #
        # ###############
        # # remove old one to reload
        # for patch in self.left_patches:
        #     patch.remove()
        # for patch in self.right_patches:
        #     patch.remove()
        # hull_left = self.pitch.convexhull(left_team_loc_dropGK[:, 0], left_team_loc_dropGK[:, 1])
        # self.left_patches = self.pitch.polygon(hull_left, ax=self.ax, edgecolor='cornflowerblue', facecolor='blue',
        #                                        alpha=0.3)
        # hull_right = self.pitch.convexhull(right_team_loc_dropGK[:, 0], right_team_loc_dropGK[:, 1])
        # self.right_patches = self.pitch.polygon(hull_right, ax=self.ax, edgecolor='orange', facecolor='red', alpha=0.3)

        # ################
        # # Draw line-up #
        # ################
        # if v11:
        #     k_lines = 3

        #     '''K-Means detect 3 line'''
        #     # centers_left, labels_left = self.cluster_pressureLine(left_team_loc_dropGK[:, 0], k=k_lines)
        #     # centers_right, labels_right = self.cluster_pressureLine(right_team_loc_dropGK[:, 0], k=k_lines)
        #     # time_2 = time.time()
        #     # time_list['kmeans'] = time_2 - time_1
        #     # time_1 = time_2

        #     '''By signal from GRF'''
        #     map_roleId2line = {0: -1, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 2}
        #     labels_left = np.array([map_roleId2line[roleId] for roleId in left_team_roles])
        #     labels_right = np.array([map_roleId2line[roleId] for roleId in right_team_roles])

        #     # set_data to reload
        #     for i in range(k_lines):
        #         line_group = np.where(labels_left == i)
        #         group_loc = left_team_loc[line_group]
        #         lines_left = self.lines_group(i, group_loc, team='left')
        #     for i in range(k_lines):
        #         line_group = np.where(labels_right == i)
        #         group_loc = right_team_loc[line_group]
        #         lines_right = self.lines_group(i, group_loc, team='right')
        #     del lines_left

        # ##########################
        # # Visualize last passing #
        # ##########################
        # if last_actions is not None:
        #     if 9 in last_actions or 10 in last_actions or 11 in last_actions:
        #         self.info_text.set_text(s='The blue team is PASSING')

        plt.tight_layout()
        self.fig.canvas.draw()
        img = PIL.Image.frombytes('RGB', self.fig.canvas.get_width_height(), self.fig.canvas.tostring_rgb())

        return img

    def __call__(self, obs):
        if len(obs['left_team'].shape) == 3:
            batch_size = obs['left_team'].shape[0]
            images = []
            skills = []
            for i in range(batch_size):
                # Extract relevant fields from obs
                left_team = obs['left_team'][i]  
                left_team_roles = obs['left_team_roles'][i]
                right_team = obs['right_team'][i]
                right_team_roles = obs['right_team_roles'][i]
                ball = obs['ball'][i]
                skill = obs['skill'][i] if 'skill' in obs else None
                
                # Format into dictionary 
                formatted_obs = {
                    'left_team': left_team,
                    'left_team_roles': left_team_roles, 
                    'right_team': right_team,
                    'right_team_roles': right_team_roles,
                    'ball': ball,
                    'skill': skill
                }
                
                # Call draw_from_obs on formatted obs
                img = self.draw_from_obs(formatted_obs)
                images.append(img)
                skills.append(skill)
        else:
            img = self.draw_from_obs(obs)
            images.append(img)
            skills.append(skill)

        # Return batch of images and skills  
        return {'images': images, 'skills': skills}
    

if __name__ == '__main__':
    time_1 = time.time()
    drawer = RealTimeDrawer()
    drawer.pitch_set()


    obs = {
        'left_team': np.array([
            [-0.9, 0],
            [-0.2, -0.3], [-0.7, -0.1], [-0.7, 0.3], [-0.6, 0.1],
            [-0.3, -0.2], [-0.5, 0], [-0.3, 0.2],
            [0.4, -0.2], [0.4, 0], [0.4, 0.2],
        ]),
        'right_team': np.array([
            [0.9, 0],
            [0.4, 0.3], [0.4, 0.1], [0.4, -0.1], [0.4, -0.3],
            [0.3, 0.3], [0.3, 0.1], [0.3, -0.1], [0.3, -0.3],
            [-0.05, -0.11], [-0.05, 0.11]
        ]),
        'ball': np.array([0.4, 0.11]),
        'left_team_roles': np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9]),
        'right_team_roles': np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9]),
    }
    # obs2 = {
    #     'left_team': np.array([
    #         [-0.9, 0],
    #         [-0.6, -0.3], [-0.5, -0.2], [-0.7, 0.3], [-0.6, 0.1],
    #         [-0.3, -0.2], [-0.5, 0], [-0.3, 0.2],
    #         [0.1, -0.2], [0.2, 0], [0.1, 0.2],
    #     ]),
    #     'right_team': np.array([
    #         [0.9, 0],
    #         [0.6, 0.3], [0.6, 0.1], [0.6, -0.1], [0.6, -0.3],
    #         [0.3, 0.3], [0.3, 0.1], [0.3, -0.1], [0.3, -0.3],
    #         [-0.05, -0.11], [-0.05, 0.11]
    #     ]),
    #     'ball': np.array([0.4, 0.11]),
    #     'left_team_roles': np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9]),
    #     'right_team_roles': np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9]),
    # }
    img = drawer.draw_from_obs(obs)
    # img = drawer.draw_from_obs(obs2)
    time_2 = time.time()
    print(f'1 frame: total time {time_2 - time_1:.4f} s')