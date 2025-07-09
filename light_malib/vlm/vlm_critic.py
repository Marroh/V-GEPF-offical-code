import base64
import io
import json
import logging
import os
import time
import sys
sys.path.insert(0, "path of ./light_malib")

import clip
import imageio
from fastapi import FastAPI
from starlette.requests import Request

import requests
import ray

import PIL
import numpy as np
import requests
from PIL import Image
from ray import serve
import transformers

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from transformers import pipeline, AutoModel, AutoTokenizer
from decord import VideoReader, cpu    # pip install decord

# from .VideoLLaMA2.videollama2.conversation import conv_templates
# from .VideoLLaMA2.videollama2.constants import DEFAULT_MMODAL_TOKEN, MMODAL_TOKEN_INDEX
# from .VideoLLaMA2.videollama2.mm_utils import get_model_name_from_path, tokenizer_MMODAL_token, process_video, \
#     process_array_image
# from .VideoLLaMA2.videollama2.model.builder import load_pretrained_model

import cv2
from light_malib.vlm.utils import RealTimeDrawer

app = FastAPI()


#
@serve.deployment(ray_actor_options={"num_cpus": 4, "num_gpus": 1}, num_replicas='auto', )
@serve.ingress(app)
class VLMCritic:
    def __init__(self, model_path, device='cuda:0'):
        self.model_path = model_path
        self.tokenizer, self.model, self.processor, self.context_len = load_pretrained_model(model_path, None,
                                                                                             get_model_name_from_path(
                                                                                                 model_path),
                                                                                             device_map=device)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        # self.model = self.model.to(device)

    def inference(self, img, mode='image'):
        # Video Inference
        # paths = ['humanoid.mp4']
        paths = [img]
        questions = [
            'Here is a picture of a football match, our players are represented by green dots, the opposing players are represented by red dots, and the football is represented by black dots. You as a strict football evaluation expert, please be objective judgment whether our players are cooperating well. Analyze then Answer "Yes" or "No"']
        # Reply:
        # The video features a kitten and a baby chick playing together. The kitten is seen laying on the floor while the baby chick hops around. The two animals interact playfully with each other, and the video has a cute and heartwarming feel to it.
        modal_list = [mode]

        # 1. Initialize the model.
        # Base model inference (only need to replace model_path)
        conv_mode = 'llama_2'

        t0 = time.time()

        # 2. Visual preprocess (load & transform image or video).
        if modal_list[0] == 'video':
            tensor = process_video(paths[0], self.processor, self.model.config.image_aspect_ratio).to(
                dtype=torch.float16, device='cuda', non_blocking=True)
            default_mm_token = DEFAULT_MMODAL_TOKEN["VIDEO"]
            modal_token_index = MMODAL_TOKEN_INDEX["VIDEO"]
        else:
            tensor = process_array_image(paths[0], self.processor, self.model.config.image_aspect_ratio)[0].to(
                dtype=torch.float16, device='cuda', non_blocking=True)
            default_mm_token = DEFAULT_MMODAL_TOKEN["IMAGE"]
            modal_token_index = MMODAL_TOKEN_INDEX["IMAGE"]
        tensor = [tensor]

        # 3. text preprocess (tag process & generate prompt).
        question = default_mm_token + "\n" + questions[0]
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_MMODAL_token(prompt, self.tokenizer, modal_token_index, return_tensors='pt').unsqueeze(
            0).to('cuda:0')

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images_or_videos=tensor,
                modal_list=modal_list,
                do_sample=False,  # for deterministic output
                # temperature=0.2,
                max_new_tokens=256,
                use_cache=True,
            )

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        t1 = time.time()
        print(f'Inference time: {t1 - t0:.2f}s')  # about 0.7s

        return outputs

    @app.get("/score")
    async def score(self, request: Request):
        img_base64, max_reward = request.query_params["img_base64"], request.query_params["max_reward"]
        # data = base64.b64decode(data)
        img_bytes = base64.b64decode(img_base64)
        image = Image.open(io.BytesIO(img_bytes))
        img = np.array(image)

        # print(img)
        # buffer = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(self.height, self.width, 3)
        # assert len(data) == 1, f'expects 1 image, but got {len(data)} images.'
        # img = data[0]
        output = self.inference(img)[0]
        print(output)
        if 'Yes' in output or 'yes' in output:
            return max_reward
        else:
            return 0.0


class CLIPCritic:
    def __init__(self):
        self.clip, self.preprocess = clip.load("/home/hf_hub/models/CLIP/RN50.pt", device="cuda")

    def __call__(self, batch):
        image = torch.from_numpy(batch['images']).to('cuda')

        # TODO: only need a single skills
        text = clip.tokenize([batch['skills'][0]]).to('cuda')

        print(f'>> image shape: {image.shape}, text token shape: {text.shape}')

        with torch.no_grad():
            image_features = self.clip.encode_image(image)
            text_features = self.clip.encode_text(text)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        potential_reward = (image_features @ text_features.T).cpu().numpy()

        # print(f'>> reward shape: {potential_reward.shape}')

        return {'rewards': potential_reward}


class CLIPCriticWithImageCat:
    def __init__(self):
        self.clip, self.preprocess = clip.load("/home/hf_hub/models/CLIP/RN50.pt", device="cuda")

    def __call__(self, batch):
        image = batch['images'].tolist()

        sampled_image = image[::3]
        # cat multiple PIL Image into one along x axis
        cat_image = np.concatenate([np.array(img) for img in sampled_image], axis=1)

        # save image to file
        imageio.imwrite(f'imgs/tempforcatimg_{time.time()}.png', cat_image)
        
        # Convert numpy array to PIL Image
        cat_image = Image.fromarray(cat_image)
        # Convert PIL Image to tensor
        cat_image = self.preprocess(cat_image).unsqueeze(0).to('cuda')

        # Tokenize text for CLIP
        text = clip.tokenize(['the blue team shows good cooperation']).to('cuda')
        
        # print(f'>> cat_image shape: {cat_image.shape}, text shape: {text.shape}')

        with torch.no_grad():
            image_features = self.clip.encode_image(cat_image)
            text_features = self.clip.encode_text(text)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        potential_reward = (image_features @ text_features.T).cpu().numpy()[0][0]  # Extract scalar value

        rewards = np.ones(len(batch['images']), dtype=np.float32) * potential_reward
        return {'rewards': rewards}


@serve.deployment(
    ray_actor_options={"num_cpus": 1, "num_gpus": 1},
    num_replicas=1,
    logging_config={
        "log_level": "CRITICAL",
        "enable_access_log": False
    }
)
@serve.ingress(app)
class ServerMiniCPM:
    def __init__(self, model_path='openbmb/MiniCPM-Llama3-V-2_5', device='cuda'):
        transformers.logging.set_verbosity_error()

        self.model = AutoModel.from_pretrained(model_path,
                                               trust_remote_code=True,
                                               torch_dtype=torch.float16,
                                               device_map=device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model.eval()

    @app.get("/chat")
    async def chat(self, request: Request):
        # print('>> request keys: ', request.query_params.keys())
        # ray.util.pdb.set_trace()
        img_base64, question = request.query_params["img_base64"], request.query_params["question"]
        img_bytes = base64.b64decode(img_base64)
        image = Image.open(io.BytesIO(img_bytes))

        # Image.open(io.BytesIO(base64.b64decode(request.query_params["img_base64"]))).save('/home/trl/football/light_malib/vlm/fucking_test.png')

        msgs = [{'role': 'user', 'content': question}]

        res = self.model.chat(
            image=image,
            msgs=msgs,
            tokenizer=self.tokenizer,
            sampling=True,  # if sampling=False, beam_search will be used by default
            temperature=0.1,
            # system_prompt='' # pass system_prompt if needed
        )

        return res


class LocalMiniCPM:
    def __init__(self, model_path='/home/openbmb-MiniCPM-Llama3-V-2_5', device='auto'):
        transformers.logging.set_verbosity_error()

        self.model = AutoModel.from_pretrained(model_path,
                                               trust_remote_code=True,
                                               torch_dtype=torch.float16,
                                               attn_implementation='sdpa',
                                               device_map=device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model.eval()

    def encode_video(self, video_path):
        MAX_NUM_FRAMES = 10

        def uniform_sample(l, n):
            gap = len(l) / n
            idxs = [int(i * gap + gap / 2) for i in range(n)]
            return [l[i] for i in idxs]

        vr = VideoReader(video_path, ctx=cpu(0))
        sample_fps = round(vr.get_avg_fps() / 1)  # FPS
        frame_idx = [i for i in range(0, len(vr), sample_fps)]
        if len(frame_idx) > MAX_NUM_FRAMES:
            frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
        frames = vr.get_batch(frame_idx).asnumpy()
        frames = [Image.fromarray(v.astype('uint8')) for v in frames]
        print('num frames:', len(frames))
        return frames

    def chat(self, image_list, question):
        # save a list of PIL images into a video named tmp.mp4
        # Create a video writer
        image_list = [np.array(img) for img in image_list]
        with imageio.get_writer('./temp.mp4', fps=25) as writer:
            for image in image_list:
                # Convert PIL Image to NumPy array
                frame = imageio.core.util.Array(image)
                # Append the frame to the video
                writer.append_data(frame)

        frames = self.encode_video('./temp.mp4')

        msgs = [
            {'role': 'user', 'content': frames + [question]},
        ]

        params = {}
        params["use_image_id"] = False
        params["max_slice_nums"] = 2  # use 1 if cuda OOM and video resolution > 448*448
        # print('Message', msgs)
        res = self.model.chat(
            image=None,
            msgs=msgs,
            tokenizer=self.tokenizer,
            **params
        )
        # print('Response:', res)
        return res


class MiniCPMCritic:
    def __init__(self):
        # Initialize MiniCPM model
        self.model = AutoModel.from_pretrained(
            '/home/hf_hub/models/openbmb-MiniCPM-o-2_6',
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation='sdpa',
        )
        self.model = self.model.eval().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(
            '/home/hf_hub/models/openbmb-MiniCPM-o-2_6', 
            trust_remote_code=True,
            use_fast=False
        )

    def process_video(self, frames):
        """Process video frames and return reward based on cooperation analysis"""
        # Prepare the question about team cooperation
        question = "This is a football match between the Red and Blue teams. The black dot is the balls. First,analyze the Blue(left) and Red(right) teams' strategies(pass, shoot, formation,etc.), then output 'True' if it does, otherwise output 'False'."
        
        # Prepare messages for the model
        msgs = [
            {'role': 'user', 'content': frames + [question]}
        ]

        # Set decode params for video processing
        params = {
            "use_image_id": False,
            "max_slice_nums": 2,  # use 1 if CUDA OOM and video resolution > 448*448
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 100,
            "repetition_penalty": 1.1,
            "max_new_tokens": 512,
            "do_sample": True,            
        }

        try:
            # Get model response
            response = self.model.chat(
                msgs=msgs,
                tokenizer=self.tokenizer,
                **params
            )
            
            # Check if response contains "true" (case insensitive)
            if 'true' in response.lower():
                return 1.0
            return 0.0
            
        except Exception as e:
            print(f"Error processing video: {e}")
            return 0.0

    def __call__(self, batch):
        frames = batch['images'].tolist()
        if batch['chosen'].tolist()[0] == 0:
            return {'rewards': np.zeros(len(frames), dtype=np.float32)}
        # 采样
        frames = frames[::3]

        # TODO: 视频保存下来看看长度是否合理
        # Get reward based on cooperation analysis
        reward = self.process_video(frames)

        # save [PIL, PIL, ...] to video
        if not os.path.exists('./temp.mp4'):
            frames = [np.array(img) for img in frames]
            with imageio.get_writer('./temp.mp4', fps=25) as writer:
                for image in frames:
                    frame = imageio.core.util.Array(image)
                    writer.append_data(frame)

        # append reward to batch
        rewards = np.ones(len(batch['images']), dtype=np.float32) * reward
            
        return {"rewards": rewards}


class MiniCPMCriticWithSampling:
    def __init__(self, ratio=0.1):
        """
        Initialize the MiniCPM critic with sampling capability.
        
        Args:
            ratio (float): Probability of processing a batch (between 0 and 1)
        """
        self.ratio = ratio
        self.use_critic = np.random.random() <= self.ratio
        # print('='*50)
        # print(f'>> use_critic: {self.use_critic}')
        # print('='*50)
        
        # Only initialize the model if we're going to use it
        if self.use_critic:
            # Initialize MiniCPM model
            self.model = AutoModel.from_pretrained(
                '/home/hf_hub/models/openbmb-MiniCPM-o-2_6',
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                attn_implementation='sdpa',
            )
            self.model = self.model.eval().cuda()
            self.tokenizer = AutoTokenizer.from_pretrained(
                '/home/hf_hub/models/openbmb-MiniCPM-o-2_6', 
                trust_remote_code=True,
                use_fast=False
            )

    def process_video(self, frames):
        """Process video frames and return reward based on cooperation analysis"""
        if not self.use_critic:
            return 0.0
            
        # Prepare the question about team cooperation
        question = "This is a football match between the Red and Blue teams. The black dot is the balls. First,analyze the Blue(left) team's strategy(pass, shoot, formation,etc.), then output 'True' if it does, otherwise output 'False'."
        
        # Prepare messages for the model
        msgs = [
            {'role': 'user', 'content': frames + [question]}
        ]

        # Set decode params for video processing
        params = {
            "use_image_id": False,
            "max_slice_nums": 1  # use 1 if CUDA OOM and video resolution > 448*448
        }

        try:
            # Get model response
            response = self.model.chat(
                msgs=msgs,
                tokenizer=self.tokenizer,
                **params
            )
            
            # Check if response contains "true" (case insensitive)
            if 'true' in response.lower():
                return 1.0
            return 0.0
            
        except Exception as e:
            print(f"Error processing video: {e}")
            return 0.0

    def __call__(self, batch):
        """
        Process a batch of frames with sampling based on the ratio.
        
        Args:
            batch (dict): A dictionary containing 'images' key with a list of frames
            
        Returns:
            dict: A dictionary with 'rewards' key containing the rewards
        """
        # If we decided not to use the critic at init time, return zeros
        if not self.use_critic:
            return {'rewards': np.zeros(len(batch['images']), dtype=np.float32), 
                    'if_sampled': np.zeros(len(batch['images']), dtype=np.float32),}
        
        frames = batch['images'].tolist()
        
        # Sample frames to reduce processing time
        frames = frames[::3]  # Take every 3rd frame

        # Process the video and get reward
        reward = self.process_video(frames)
        
        # Create rewards array with the same reward for all frames
        rewards = np.ones(len(batch['images']), dtype=np.float32) * reward
            
        return {"rewards": rewards, 
                "if_sampled": np.ones(len(batch['images']), dtype=np.float32),}


if __name__ == '__main__':
    pass