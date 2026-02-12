<h1 align="center">
	SeedPolicy: Horizon Scaling via Self-Evolving Diffusion Policy for Robot Manipulation
</h1>

<div align="center">

Youqiang Gui<sup>\*</sup>, Yuxuan Zhou<sup>\*</sup>, Shen Cheng, Xinyang Yuan, Haoqiang Fan, Peng Cheng, Shuaicheng Liu


[arXiv](https://arxiv.org/abs/23xx.xxxxx)

</div>

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
<div align="center">
    <i>Figure 2: Illustration of SEGA. (Add your detailed caption here describing the specific module...)</i>
</div>

<br>

We present **SeedPolicy**, a novel framework that leverages self-evolving mechanisms to scale the horizon of diffusion policies for enhanced robotic manipulation capabilities.

# 🛠️ Installation

Our installation process follows the **RoboTwin** platform standards. 

Please refer to the [RoboTwin Official Documentation](https://robotwin-platform.github.io/doc/index.html) for detailed instructions on:
1.  **Environment Setup**: Setting up the python environment and dependencies.
2.  **Data Collection**: Collecting expert demonstrations.
3.  **Data Processing**: Processing data for Policy/Diffusion Policy (DP) training.

# 🧑🏻‍💻 Usage

## 1. Step One
(Add description for step 1 here)

## 2. Step Two
(Add description for step 2 here)

## 3. Step Three
(Add description for step 3 here)

# 👍 Citation
If you find our work useful, please consider citing:

```bibtex
@article{seedpolicy202x,
  title={SeedPolicy: Horizon Scaling via Self-Evolving Diffusion Policy for Robot Manipulation},
  author={Gui, Youqiang and Zhou, Yuxuan and Cheng, Shen and Yuan, Xinyang and Fan, Haoqiang and Cheng, Peng and Liu, Shuaicheng},
  journal={arXiv preprint arXiv:23xx.xxxxx},
  year={202x}
}
