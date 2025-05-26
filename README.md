<div align="center">

<img src="figs/logo.png" width="260"/>

## 🚀 Online RL with Simple Reward Enables Training VLA Models with Only One Trajectory

</div>

<!-- <div align="center">
  <p>
    <a href="#news" style="text-decoration: none; font-weight: bold;">🎉 News</a> •
    <a href="#overview" style="text-decoration: none; font-weight: bold;">📖 Overview</a> •
    <a href="#main-results" style="text-decoration: none; font-weight: bold;">📃 Main Results</a> •
    <a href="#getting-started" style="text-decoration: none; font-weight: bold;">✨ Getting Started</a>
  </p>
  <p>
    <a href="#acknowledgement" style="text-decoration: none; font-weight: bold;">🌻 Acknowledgement</a> •
    <a href="#contact" style="text-decoration: none; font-weight: bold;">📨 Contact</a> •
    <a href="#todo" style="text-decoration: none; font-weight: bold;">📝 TODO</a> •
    <a href="#citation" style="text-decoration: none; font-weight: bold;">🎈 Citation</a>
  </p>
</div> -->

With **only one trajectory** per task for supervised fine-tuning (SFT), SimpleVLA-RL leverages online reinforcement learning (RL) with simple outcome-level 0/1 rule-based reward signals to achieve performance comparable to full-trajectory SFT.

<div align="center">
<img src="figs/teaser.png" alt="Performance of SimpleVLA-RL." width="75%" />

<sub>*Our openvla-oft model design differs from the official one. Our setup: third-person image, language instruction; parallel decoding (PD) & action chunking (AC).Official setup: third-person image, wrist camera image, robot proprioceptive state, language instruction; PD, AC, and continuous actions with L1 regression (Cont-L1).*</sub>
</div>

# 🎉News

- **[2025-05-26]** We release the code of **SimpleVLA-RL**.

# 📖Overview

We introduce SimpleVLA-RL, a simple yet effective approach for online Reinforcement Learning (RL) on Vision-Language-Action (VLA) models, which utilizes only outcome-level 0/1 rule-based reward signals directly obtained from simulation environments.

<div align="center">
<img src="figs/vla-rl-1.png" alt="Overview of SimpleVLA-RL." width="75%" />
</div>


# 📃Main Results
We evaluate SimpleVLA-RL on the LIBERO using OpenVLA-OFT.

SimpleVLA-RL improves the performance of OpenVLA-OFT to **97.6 points** on LIBERO-Long and sets a new state-of-the-art. Remarkably, using only one trajectory per task for cold-start SFT, SimpleVLA-RL raises the performance of OpenVLA-OFT from 17.3 to 91.7, yielding an improvement of **74.4 points (430.1%)**.

<div align="center">
<img src="figs/main.png" alt="Main Results of SimpleVLA-RL." width="75%" />
</div>

# ✨ Getting Started

## 1. Environment Setup

Our code is built on top of [veRL](https://verl.readthedocs.io/en/latest/start/install.html). To begin, install veRL and configure the environment for the specific Vision-Language-Action (VLA) model you intend to use. The following illustrates how to set up OpenVLA-OFT.

1. **Install veRL**
   Follow the installation guide for veRL [here](https://verl.readthedocs.io/en/latest/start/install.html).
2. **Install OpenVLA-OFT**
   Set up OpenVLA-OFT by following the instructions in the [OpenVLA-OFT repository](https://github.com/moojink/openvla-oft).

---


## 2. Prepare the SFT Model

An SFT model is required to initiate RL training.

* **Pre-trained OpenVLA-OFT models** can be downloaded [here](https://huggingface.co/collections/Haozhan72/simplevla-rl-6833311430cd9df52aeb1f86), including:
  * `libero-10 traj1 SFT`
  * `libero-10 trajall SFT`
* **Pre-trained OpenVLA models** can be downloaded [here](https://huggingface.co/openvla).
* For other models, you’ll need to fine-tune them yourself.

---

## 3. Running RL

Before executing the training script, ensure the following configurations are updated:

1. **Add your Weights and Biases (WanDB) API Key**
   Update the `WANDB_API_KEY` field in `simpleVLA-RL/scripts/align.json` with your WanDB key.
2. **Modify Key Variables**
   Update the following fields in `run_openvla_oft_rl.sh`:
   * `WANDB_API_KEY`: Your WanDB API key.
   * `EXPERIMENT_NAME`: Name of your experiment.
   * `SFT_MODEL_PATH`: Path to your SFT model.
   * `CKPT_PATH`: Path to save checkpoints.
   * `DATASET_NAME`: Name of the dataset.
   * `ALIGN_PATH`: Path to your `align.json` file.
3. **Run RL**
   Run the following command to run RL for OpenVLA-OFT on the LIBERO benchmark with 8 * NVIDIA A800 80GB GPUs:
   ```bash
   bash examples/run_openvla_oft_rl.sh
   ```

---

## 4. Evaluating RL Results

To evaluate your model's performance, simply enable evaluation mode by setting `trainer.val_only=True` in the training configuration. Then, run the same script:

```bash
bash examples/run_openvla_oft_rl.sh
```


# 🌻Acknowledgement

We develop this preview version of the code based on [veRL](https://github.com/volcengine/verl), [OpenVLA-OFT](https://github.com/moojink/openvla-oft), and [PRIME](https://github.com/PRIME-RL/PRIME). We acknowledge their significant contributions!
For further details and updates, please refer to the official documentation and repositories of the respective projects.

# 📨Contact

- Haozhan Li: zhan72426@gmail.com
- Ning Ding: dingning@mail.tsinghua.edu.cn

# 📝TODO

* ​**Models**​:
  * ✅ Support OpenVLA and OpenVLA-OFT
  * ⏳ Support Pi0 fast tokenizer
* ​**Benchmarks**​:
  * ✅ Support LIBERO benchmark
  * ⏳ Support RoboTwin benchmark

# 🎈Citation

If you find SimpleVLA-RL helpful, please cite us.
```bibtex
@misc{li2025simplevlarl,
  title={SimpleVLA-RL: Online RL with Simple Reward Enables Training VLA Models with Only One Trajectory},
  author={SimpleVLA-RL Team},
  year={2025},
  howpublished={\url{https://github.com/}},
  note={Github Repository},
}
```
