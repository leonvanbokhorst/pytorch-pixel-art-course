# PyTorch Pixel Art Deep Learning Course

## Overview

Welcome! This is a **9-day, hands-on PyTorch course** designed for AI engineering students (and curious creators!) who know Python and some ML basics but haven't built models from scratch. The goal is to build a solid foundation in PyTorch's core components (tensors, autograd, neural network modules, data loading, training loops) **specifically applied to the domain of pixel art**. We'll explore how deep learning can generate, enhance, and interact with pixel art, preparing you for advanced creative AI topics.

- **Structure:** Modular and self-paced over 9 conceptual days, organized into directories in this repository. Each day builds PyTorch knowledge using pixel art examples.
- **Learning:** Hands-on experimentation and reflection are key. Run the code, tweak parameters, generate some pixels, and see what happens!
- **Prerequisites:** Basic Python, NumPy familiarity, high-level ML concepts (what training a model means), and an appreciation for blocky aesthetics! 👾

## Installation

First, ensure you have Python installed (version 3.8 or higher is recommended).

We recommend using `uv` for faster dependency management and virtual environment creation. If you don't have `uv` installed, follow the official installation guide: [https://github.com/astral-sh/uv#installation](https://github.com/astral-sh/uv#installation)

1. **Create and activate a virtual environment:**

   ```bash
   # Create the environment (e.g., named .venv)
   uv venv
   # Activate it (syntax depends on your shell)
   # Linux/macOS (bash/zsh)
   source .venv/bin/activate
   # Windows (cmd.exe)
   .venv\Scripts\activate.bat
   # Windows (PowerShell)
   .venv\Scripts\Activate.ps1
   ```

2. **Install dependencies:**

   ```bash
   uv pip install .  # Installs dependencies from pyproject.toml
   # Check the official PyTorch website (https://pytorch.org/get-started/locally/)
   # for CUDA-specific versions if you have an NVIDIA GPU and update pyproject.toml accordingly.
   ```

Using an interactive environment like Jupyter Notebook or Google Colab is highly recommended for experimentation. Make sure to install `ipykernel` (`uv pip install ipykernel`) in your virtual environment and select the correct kernel (`.venv`) if using Jupyter.

## 🚀 Our Epic Quest: From Pixel to Capstone!

Welcome, brave adventurer, to a grand quest! We're not just learning PyTorch and wrangling pixels; we're embarking on a journey through the treacherous, yet rewarding, lands of Deep Learning.

**The Premise:** Each day, or module, of this course represents a stage in your journey. You'll start as a humble Pixel Paladin, armed with only basic Python knowledge, and progressively tackle challenges, learn new spells (algorithms!), forge powerful artifacts (models!), and ultimately, face the final Capstone Citadel.

**Why a Story?**

1.  **Context is Key:** Understanding _why_ you're learning `torch.nn.Linear` is easier when you know it's the key to unlocking the Gates of Neuron.
2.  **Engagement++:** Let's face it, fighting `RuntimeError` dragons is more fun than just debugging.
3.  **Memory Palace... sorta:** Associating concepts with story beats helps things stick!

**The Journey Ahead (A Rough Sketch):**

- **Days 1-X (The Shifting Sands of Setup):** Navigating the environment, installing libraries (gathering supplies!), understanding tensors (learning the basic runes).
- **Days X-Y (The Forest of Fundamentals):** Basic image manipulation, simple neural networks (first encounters with friendly slimes... I mean, models).
- **Days Y-Z (The Mountains of Mastery):** Convolutional Neural Networks, advanced techniques (battling complex beasts, scaling treacherous peaks).
- **The Capstone Citadel:** The final project! Combine all your skills to build something magnificent.

Each day's `README` will set the scene for that stage of your quest. Expect challenges, triumphs, and maybe a few eccentric wizards along the way. Now, sharpen your wits (and your code editor), and let the adventure begin!

---

## 📚 Course Structure

Think of this course like building a pixel art creation tool (or maybe enchanting a sprite!). Each day adds a crucial component, building on the last:

1. **Day 1 ([`day_01_intro_tensors`](./day_01_intro_tensors/)):** Basic materials (Tensors - representing pixel colors, coordinates, or entire sprites).
2. **Day 2 ([`day_02_tensor_ops`](./day_02_tensor_ops/)):** Shaping the materials (Tensor operations - manipulating colors, scaling sprites, basic filters).
3. **Day 3 ([`day_03_autograd`](./day_03_autograd/)):** The magic learning wand (Automatic differentiation - the core of how models learn pixel patterns).
4. **Day 4 ([`day_04_nn_module`](./day_04_nn_module/)):** Blueprints for pixel magic (`nn.Module` for building models/layers that understand pixel art).
5. **Day 5 ([`day_05_datasets_dataloaders`](./day_05_datasets_dataloaders/)):** Loading our pixel palettes (`Dataset`, `DataLoader` for feeding pixel art data).
6. **Day 6 ([`day_06_training_loop`](./day_06_training_loop/)):** Training our pixel generator! (The training loop - putting it all together to learn from pixel data).
7. **Day 7 ([`day_07_evaluation_improvement`](./day_07_evaluation_improvement/)):** Quality control and style refinement (Evaluation, fixing artifacts, saving models and generated art).
8. **Day 8 ([`day_08_gpu_performance`](./day_08_gpu_performance/)):** Super-fast rendering! (Using GPUs for speedier pixel generation/training).
9. **Day 9 ([`day_09_capstone_project`](./day_09_capstone_project/)):** Grand Pixel Showcase! (Putting it all together in an adaptable pixel art project template).

The concepts build progressively, so it's best to work through the days sequentially.

## Learning PyTorch Through Pixel Art

This course uses the fun and visual domain of pixel art to teach PyTorch fundamentals. Instead of generic examples, we'll directly apply concepts to tasks like:

- Representing pixel sprites and palettes as tensors.
- Using tensor operations for simple image manipulations (color swapping, flipping).
- Building simple neural networks to generate small pixel patterns.
- Training models on datasets of existing pixel art.
- Evaluating the quality and style of generated pixel art.

The goal is to make learning PyTorch engaging and immediately applicable to a creative domain. As you go through each day:

1.  **Understand the Core Concept:** Grasp the fundamental PyTorch idea being introduced (e.g., what `autograd` does).
2.  **See it Applied to Pixels:** Examine the provided code examples to see how the concept translates to pixel art tasks.
3.  **Experiment!** Modify the code. Change the input data (maybe load your own small pixel art?). Adjust model parameters. See how the output pixels change. What happens if you train for longer? What if you use a different color palette?
4.  **(Optional) Chat with an AI:** If you get stuck or want to explore variations, ask an AI assistant (like me!) for help. "How could we modify this Day 4 model to generate 16x16 sprites instead of 8x8?" or "Can you explain this loss function in the context of pixel color accuracy?"

By the end, you'll not only understand PyTorch fundamentals but also have a repository filled with code demonstrating how to apply deep learning to the charming world of pixel art!
