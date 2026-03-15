<h1 align="center">
	SeedPolicy: Horizon Scaling via Self-Evolving Diffusion Policy for Robot Manipulation
</h1>


# 📚 Overview

![](./files/framework.png)
<div align="left">
    <i>
    <b>Overview of the SeedPolicy framework.</b> 
    The system takes current RGB images and joint poses as input, encoding them via a ResNet Encoder. 
    The core <b>Self-Evolving Gated Attention (SEGA)</b> module (blue box) recursively updates a time-evolving latent state (<i>State t</i>) to capture long-term spatiotemporal dependencies while generating enhanced observation features (<i>EObs<sub>t</sub></i>). 
    These context-rich features are then fed into the Action Expert, a transformer-based diffusion model, to predict a sequence of future actions.
    </i>
</div>

<br>


![](./files/SEGA.png)
<div align="left">
    <i>
    (a) SEGA employs a dual-stream design: the <b>State Update</b> stream (top) evolves the latent state (<i>State<sub>t-1</sub></i>) by integrating new observations, while the <b>State Retrieval</b> stream (bottom) utilizes historical context to generate enhanced observation features (<i>EObs<sub>t</sub></i>).
    <br>
    (b) The <b>Self-Evolving Gate (SEG)</b> dynamically computes a gating signal directly from the cross-attention maps. It selectively fuses the intermediate evolved state (Inter &middot; <i>S<sub>t</sub></i>) with the previous state, ensuring only semantically relevant information is preserved while filtering out noise.
    </i>
</div>
<br>



# 🛠️ Installation

Our installation process follows the **RoboTwin** platform standards. 

Please refer to the [RoboTwin Official Documentation](https://robotwin-platform.github.io/doc/index.html) for detailed instructions on:
1.  **Environment Setup**: Setting up the python environment and dependencies.
2.  **Data Collection**: Collecting expert demonstrations.
3.  **Data Processing**: Processing data for Policy/Diffusion Policy (DP) training.

# 🤗 Pre-trained Models

We provide pre-trained model checkpoints for the three typical tasks highlighted in our paper. You can download them directly from our Hugging Face repository:

👉 [**SeedPolicy Model Checkpoints on Hugging Face**](https://huggingface.co/guiyouqiang/SeedPolicy/tree/main)


# 🧑🏻‍💻 Usage

## 1. Train Policy
```
bash train.sh ${task_name} ${task_config} ${expert_data_num} ${seed} ${action_dim} ${gpu_id} ${config_name}

# Example:
# bash train.sh beat_block_hammer demo_clean 50 0 14 0 train_diffusion_transformer_hybrid_workspace
```
## 2. Eval Policy
```
bash eval.sh ${task_name} ${task_config} ${ckpt_setting} ${expert_data_num} ${seed} ${gpu_id} ${config_name} ${timestamp}

# Example 1: Standard Evaluation
# bash eval.sh beat_block_hammer demo_clean demo_clean 50 0 0 train_diffusion_transformer_hybrid_workspace "'20260106-143723'"
# This command uses the policy trained on the `demo_clean` setting ($ckpt_setting)
# and evaluates it using the same `demo_clean` setting ($task_config).

# Example 2: Generalization Evaluation
# To evaluate a policy trained on the `demo_clean` setting and tested on the `demo_randomized` setting, run:
# bash eval.sh beat_block_hammer demo_randomized demo_clean 50 0 0 train_diffusion_transformer_hybrid_workspace "'20260106-143723'"
```
The evaluation results, including videos, will be saved in the eval_result directory under the project root.


# 😺 Acknowledge
Our code is generally built upon: [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) and [RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin). Specifically, the implementation of our state update code references [CUT3R](https://github.com/CUT3R/CUT3R) and [TTT3R](https://github.com/Inception3D/TTT3R). We thank all these authors for their nicely open sourced code and their great contributions to the community.


