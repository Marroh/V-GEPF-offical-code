# V-GEPF: Vision-Based Generic Potential Function for Policy Alignment in Multi-Agent Reinforcement Learning

[![license](https://img.shields.io/badge/license-MIT-blue.svg?style=flat)](./LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)]()

This repository contains the official implementation of the AAAI 2025 paper "Vision-Based Generic Potential Function for Policy Alignment in Multi-Agent Reinforcement Learning". This work introduces **V-GEPF**, a novel framework that leverages Vision-Language Models (VLMs) to generate potential-based rewards for complex multi-agent reinforcement learning tasks in the Google Research Football (GRF) environment.

This project is built upon the [GRF MARL Lib](https://github.com/jidiai/GRF_MARL) and extends it with VLM-driven reward shaping and policy alignment.

----

## Contents
- [Install](#install)
- [How to Run V-GEPF](#how-to-run-v-gepf)
- [Core Implementation](#core-implementation)
- [Alternative: xT-based Potential Reward](#alternative-xt-based-potential-reward)
- [Original GRF MARL Lib Documentation](#original-grf-marl-lib-documentation)
- [Citation](#citation)
- [Contact](#contact)

----

## Install

The installation largely follows the setup of the base [GRF MARL Lib](https://github.com/jidiai/GRF_MARL). Additionally, this project requires dependencies for the MiniCPM-o model.

1.  **Create a Conda Environment**
    You can use any tool to manage your python environment. Here, we use conda as an example.
    ```bash
    conda create -n v-gepf python==3.9
    conda activate v-gepf
    ```

2.  **Install Google Research Football**
    Follow the instructions in the [official repo](https://github.com/google-research/football) to install the GRF environment.

3.  **Install Base Framework**
    ```bash
    pip install . -e
    ```

4.  **Install MiniCPM-o Dependencies**
    V-GEPF relies on the MiniCPM-o vision-language model. Please follow the installation guide at the [official MiniCPM-o repository](https://github.com/OpenBMB/MiniCPM-o) to set up the necessary components.

----

## How to Run V-GEPF

### 1. Configuration

Before running the experiments, you need to configure the paths to your downloaded VLM and CLIP models.

-   **Set Model Paths:**
    Open `light_malib/vlm/vlm_critic.py` and update the hardcoded paths for the models you intend to use. Pay attention to the `__init__` methods of classes like `LocalMiniCPM`, `MiniCPMCritic`, and `CLIPCritic`. For example:
    ```python
    class LocalMiniCPM:
        def __init__(self, model_path='/path/to/your/openbmb-MiniCPM-Llama3-V-2_5', device='auto'):
            # ...

    class CLIPCritic:
        def __init__(self):
            self.clip, self.preprocess = clip.load("/path/to/your/CLIP/RN50.pt", device="cuda")
    ```

-   **(Optional) Set OpenAI API Keys:**
    If you plan to use GPT series of models, as the VLM, open `light_malib/vlm/utils.py` and set your OpenAI API key and base URL:
    ```python
    os.environ['OPENAI_API_KEY'] = 'your-api-key'
    os.environ["OPENAI_BASE_URL"] = 'your-base-url'
    ```

### 2. Execution

Run an experiment by executing one of the shell scripts located in the `/scripts` directory. These scripts are pre-configured for different scenarios.

For example, to run V-GEPF with MAPPO in the 11-vs-11 full-game scenario:
```bash
sh scripts/run_vllm_mappo_11v11.sh
```

----

## Core Implementation

The main logic for V-GEPF is located in the following files:

-   `light_malib/vlm/vlm_critic.py`: Implements the VLM-based critics (e.g., `MiniCPMCritic`, `CLIPCritic`) that analyze game states and generate potential rewards.
-   `light_malib/vlm/utils.py`: Contains helper utilities, including `RealTimeDrawer` for converting game states into images and `vLLMAgent` + `vLLMMemory` for high-level skill selection.
-   `light_malib/algorithm/mappo/trainer.py`: Integrates potential rewards from the VLM critic into the MAPPO training loop, guiding policy learning toward desired behaviors. Supports efficient batch rendering and VLM inference, leveraging Ray for scalable parallelization.

----

## Alternative: xT-based Potential Reward

This repository also includes a traditional, non-VLM potential reward function based on Expected Threat (xT). To enable it:

1.  Open the file `light_malib/envs/gr_football/env.py`.
2.  In the `step` method, uncomment the code block responsible for calculating and adding the `xG` and `xT` potential rewards.

----

## Original GRF MARL Lib Documentation

For more information about the underlying MARL framework, including its architecture, supported algorithms (IPPO, MAPPO, HAPPO), and other features, please refer to the original `README.md` content available in the [GRF MARL Lib repository](https://github.com/jidiai/GRF_MARL) or in the `README_GRF_MARL_Lib.md` file in this repo.

----

## Citation

If you use this code in your research, please consider citing our paper:

```bibtex
@inproceedings{ma2025vision,
  title={Vision-Based Generic Potential Function for Policy Alignment in Multi-Agent Reinforcement Learning},
  author={Ma, Hao and Wang, Shijie and Pu, Zhiqiang and Zhao, Siyao and Ai, Xiaolin},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={18},
  pages={19287--19295},
  year={2025}
}
```
