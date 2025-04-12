# PyTorch Fundamentals Course

## Overview

Welcome! This is a **9-day, domain-agnostic PyTorch course** designed for AI engineering students who know Python and some ML basics but haven't built models from scratch. The goal is to build a solid foundation in PyTorch's core components (tensors, autograd, neural network modules, data loading, training loops), preparing you for advanced deep learning topics.

- **Structure:** Modular and self-paced over 9 conceptual days, organized into directories in this repository.
- **Learning:** Hands-on experimentation and reflection are key. Run the code, change it, see what happens!
- **Prerequisites:** Basic Python, NumPy familiarity, high-level ML concepts (what training a model means).

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

## Course Structure: How the Days Connect

Think of this course like building a robot (or baking a complex cake!). Each day adds a crucial component, building on the last:

1. **Day 1 ([`day_01_intro_tensors`](./day_01_intro_tensors/)):** Basic materials (Tensors - the nuts and bolts or flour/eggs).
2. **Day 2 ([`day_02_tensor_ops`](./day_02_tensor_ops/)):** Shaping the materials (Tensor operations - welding or mixing).
3. **Day 3 ([`day_03_autograd`](./day_03_autograd/)):** The magic brain wiring (Automatic differentiation - the learning mechanism).
4. **Day 4 ([`day_04_nn_module`](./day_04_nn_module/)):** Blueprints for assembly (`nn.Module` for building models/layers).
5. **Day 5 ([`day_05_datasets_dataloaders`](./day_05_datasets_dataloaders/)):** Fuel/ingredients supply chain (`Dataset`, `DataLoader` for feeding data).
6. **Day 6 ([`day_06_training_loop`](./day_06_training_loop/)):** Ignition sequence! (The training loop - putting it all together to learn/bake).
7. **Day 7 ([`day_07_evaluation_improvement`](./day_07_evaluation_improvement/)):** Quality control and testing (Evaluation, fixing issues, saving results).
8. **Day 8 ([`day_08_gpu_performance`](./day_08_gpu_performance/)):** Turbo boost! (Using GPUs for speed).
9. **Day 9 ([`day_09_capstone_project`](./day_09_capstone_project/)):** Grand Finale! (Putting it all together in an adaptable project template).

The concepts build progressively, so it's best to work through the days sequentially.

## Adapting Examples to Your Domain (with AI Assistance!)

This course covers domain-agnostic PyTorch fundamentals. The real fun begins when you apply these concepts to something _you_ care about (like guitar pedals, astrophysics, cat behavior analysis, you name it!).

Here's how you can use this repository and an AI assistant (like Cursor, which powered this interaction!) to create domain-specific examples:

1. **Clone this Repository:** Start with a fresh copy of this course structure.
2. **Choose Your Passion:** Decide on the domain or idea you want to infuse into the examples (e.g., guitar effects, predicting stock prices, classifying plant species).
3. **Ask the AI to adapt the examples to your domain:** Ask the AI to help you create examples in your domain. Example prompt: _"Create examples for all days of the course in the domain of [your domain], using the concepts learned in each day."_
4. **Go Day-by-Day:** As you approach each day's topic:
   - Understand the **core PyTorch concept** (e.g., what a Tensor is, how `nn.Module` works).
   - **Chat with your AI assistant!** Ask things like:
     - "For Day 1 (Tensors), how could I represent [your domain data, e.g., guitar pedal knob settings] as tensors?"
     - "For Day 4 (nn.Module), can you help me sketch a _simple_ model that takes [your domain input, e.g., pedal settings] and predicts [your domain output, e.g., perceived 'fuzziness']?"
     - "Can we update the example code in `day_X/some_script.py` to use [your domain variables] instead of the generic ones?"
   - **Collaborate on Code:** Work with the AI to modify the example code snippets (or create new ones) in each day's directory to reflect your domain.
5. **Keep it Simple (at first):** Especially early on, focus on applying the _PyTorch concept_ in your domain's context, even if the model or data isn't perfectly realistic yet. The goal is to learn PyTorch _through_ your domain.

By the end, you'll not only understand PyTorch fundamentals but also have a repository filled with examples directly relevant to your interests!
