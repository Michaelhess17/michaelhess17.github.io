+++
date = '2025-03-27T11:42:17-04:00'
draft = false
title = 'Research Tools You Need to be Using'
tags = ["tools", "research", "computer science", "tutorials"]
categories = ["tools", "research", "computer science", "tutorials"]
+++
## What's missing from most research development pipelines?

Okay, I'll admit it. I'm a tooling nerd. For example, I started doing my dissertation research like many computational scientists by using Anaconda, NumPy, and Pandas, but I quickly realized the issues with this setup. First, why are so many NumPy updates not backwards compatible? I've lost count of the number of times an environment installation has gone completely haywire due to a mismatched NumPy version. These mismatches generally lead to a five-year-old Stack Overflow post where the overwhelmingly accepted answer is just "oh, run `pip install numpy==1.24.x` or whatever version fixes the one bug you're having." Who knows what other errors will come up when you install that version, though?

I've lost several days' worth of time at this point in my PhD installing, recreating, and managing environments. I am the IT administrator for our laboratory, so I've seen the [Install TensorFlow with pip](https://tensorflow.org/install/pip) page *fundamentally* change installation instructions at least five times in the last few years. It's maddening, and it needs to be fixed. Fortunately, for scientists who care about reproducibility and avoiding environment headaches, there are some unique tools that will help declaratively manage your environments, making them sharable and reproducible on basically any machine when the proper precautions are taken. There are four main tools (and two bonus tools!) that I'm going to talk about that can dramatically improve the robustness of your research pipeline, make your work easily sharable or portable, and prevent wasting time trying to get environments installed just the right way.

## Pixi -- A modern Python environment manager

I used to love Anaconda. Using it over tools like `venv` made managing complicated system dependencies like CUDA Toolkit much simpler. Anyone who has ever gone to the [CUDA Support Matrix](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/support-matrix.html) knows the pain of managing Nvidia drivers, CUDA versions, and Python package versions simultaneously. Once you get an environment that works, you pray to God that you never need to update a package or add something new. Further, you have to follow installations in a specific order, or you risk ending up in a different state than intended. If one step of the installation fails, your environment can be left in a half-updated (read: broken) state. Good luck getting your environment back to working, especially if you're using the default Anaconda solver with an older Anaconda version—you'll be waiting for the solver to determine workable installation candidates for ages.

Some of the main issues with Anaconda—mainly speed and compatibility issues—have been improved with the use of better solvers like Mamba and the conda-forge ecosystem. Honestly, even if you don't start using Pixi, I would recommend *immediately* updating your conda solver to libmamba and start using the conda-forge channel where possible. These changes will dramatically improve your quality of life. Some other conda issues like reproducibility and portability can be improved by using conda `.lock` files, which keep track of the exact package versions you use in the environment. The main problems with this approach are:

1. The lack of guarantees that the environment will be installable on any other machine than your own.
2. Managing `pip`, `conda`, and system dependencies separately still requires an installation procedure that can fail for many reasons.

Pixi solves these issues by prompting you to create your environment *declaratively*. This means that the process to build your environment and the system requirements can be spelled out directly in a configuration file. This configuration file defines package versions and system requirements for the environment, as well as which platforms you want the environment to be compatible with. If you are writing a software package like [SLEAP](https://github.com/SLEAP) that you intend to share with others, you need to make sure that all of the package dependencies will be installable on Linux, Mac, and Windows. Pixi lets you define this requirement and ensures that all installed versions of packages have binaries available for all defined platforms. This means that if you work on an Ubuntu machine at work, you can be confident that your code will still run on your Windows or Mac machine at home. Further, once the configuration file is complete, you just need to run `pixi install`, and all dependencies will be downloaded to a `.pixi` folder, which can be deleted and recreated from the configuration file whenever desired. With a warm cache, installing a new environment—even complicated ones that require setting environment variables or running tasks—can be completed in less than a few seconds.

Pixi also recently incorporated the [uv package manager](https://astral.sh/blog/uv), which handles the pip dependencies declaratively on top of the conda dependencies, making it a complete environment and package management solution for Python. UV is another tool definitely worth checking out by itself, but I have not used it much yet.

## Nix -- The declarative system package manager

The Nix package ecosystem is huge—[bigger than any other with more "fresh" packages](https://discourse.nixos.org/t/nixpkgs-has-been-the-largest-repository-for-months/10667). Nix is a powerful package manager and build system that allows for fully declarative package management. Unlike traditional package managers like `apt` or `yum`, which modify the global system state, Nix ensures that installations are fully isolated and reproducible. This means you can define an entire research computing environment in a single `flake.nix` or `default.nix` file and share it with others, ensuring that everyone has the exact same environment.

For example, here’s how you can create a simple development shell with Python and Jax using Nix:

```nix
{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
  buildInputs = [
    pkgs.python311
    pkgs.jax
  ];
}
```
Running `nix develop` in the directory containing this file will creating a fake File Hierarchy System for the development environment without actually altering the state of your system in any way, and for this example, the jax package will already be preloaded into the "global" python installation. 

The Nix operating system (NixOS) takes idea this even further, by ensuring that the build dependencies for every tool on the system are known and tractable. This comes with a few limitations that require non-traditional workarounds, but ultimately I have not been able to find anything in NixOS that I can't do somehow. Since the package configuration is declarative like Pixi, you'll never have a system in a half-updated state (again read: broken state) and dependencies can be updated on a live system. For example, on a live system, you can update the Nvidia drivers and suddenly, *you just have them running on the machine*. You can test whether those driver versions fix your issue, and if not you can just revert the upgrade and your system will be *exactly like it was before the upgrade*. Anyone who has dealt with the headache of upgrading Nvidia drivers and expecting at least broken python environments in the best case or a hang on boot in the worse case. Having the ability to test whether changes fix problems without worry drastically changes how willing I am to try out different software versions or packages. 

Even if some change does brick the machine, though, Nix keeps old versions of the configuration files, and so the system state can be restored to an older version with 0 interruption. Just select an older configuration file from the GRUB menu, and you're loaded back into a working system. Before using NixOS, I would never consider upgrades near conferences, Zoom calls, or presentations because I would worry about breaking my system, losing data, and having to start my configuration again from scratch. Now, I feel free to try new packages whenever! I am using Hyprland, a feature-rich, Wayland-based window compositor for linux, which typically is challenging to install and configure, but just seems to work for me. 


## Jax -- Device/accelerator agnostic code for rapid prototyping and performance when necessary

The landscape for scientific computing and automatic differentiation is evolving quickly. Tools like Jax provide NumPy-like operations while enabling just-in-time (JIT) compilation and automatic differentiation via XLA. This makes it ideal for high-performance computing tasks, particularly for researchers working on neural networks, differential equations, and probabilistic models.

Here’s a simple example of using Jax to compute gradients:

```python
import jax.numpy as jnp
from jax import grad

def f(x):
    return x**2 + 3*x + 2

df_dx = grad(f)
print(df_dx(2.0))  # Output: 7.0
```

Jax seamlessly allows to same code to perform computations across CPUs, GPUs, and TPUs, making it a versatile tool for researchers looking to optimize performance without rewriting their code for different hardware backends. It is straightforward to convert most NumPy code to the JAX framework, which makes parallelizing or running existing code on accelerators so easy!

## Bonus: PyMC -- Make your modeling assumptions explicit and use Bayesian modeling

PyMC is a powerful tool for Bayesian modeling that allows researchers to specify probabilistic models explicitly. Unlike traditional statistical approaches, Bayesian inference provides full posterior distributions over parameters, offering better uncertainty quantification. PyMC integrates well with Jax, allowing for faster sampling and inference. While this isn't a tool to help with environment management, it fits with the theme of spelling out our assumptions at the beginning of a research project to make later computation easier. Using the Bayesian framework requires only a little extra work, but it makes understanding, interpreting, and having confidence in your modeling a lot easier!

## Bonus: Git -- An underutilized tool in science for prototyping, code testing, and sharing

Despite its widespread use in software development, Git remains underutilized in academic research. Using Git properly can improve reproducibility, collaboration, and version control in scientific computing. Tools like GitHub Actions can even automate testing and documentation deployment, making research code more robust and shareable. In my case, I keep backups of my research project synced with Github and I can take my research with my wherever I go. Sometimes, I will not have internet access or don't want to connect to a work VPN/SSH into a work machine. Having the ability to take my code with me and always have online backups means that even in the case of catastrophic failure, my code will always be safe. Further, I keep my NixOS configuration stored in a Github repository, so if I do manage to break my operating system beyond repair or have a hard drive failure, I will only need to download my configuration files and can have my exact configuration (keyboard shortcuts, application preferences, and all) restored in a matter of minutes. Similarly, I can keep these shortcuts, preferences, and package versions consistent across multiple machines, which keeps me from wasting time trying to get environments working so I can spend more time trying to get code and models working!

