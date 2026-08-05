---
title: "Building an HPC Software Stack with Environment Modules"
categories:
  - Python 
tags:
  - HPC 
  - SLURM
  - Raspberry Pi
  - Singularity
  - MPI
header:
  image: &image "https://miro.medium.com/v2/resize:fit:750/format:webp/1*lOfQ09NChahs0hQitYSQCw.png"
  caption: ""
  teaser: *image
link: 
classes: wide
toc: false
toc_label: "Table of Contents"
# toc_icon: "cog"
author_profile: false
layout: splash
---

{: .notice--info}
In my [previous post](https://hghcomphys.github.io/slurm-hpc-cluster-with-raspberry-pis/),
, I explained the steps for building a SLURM High-Performance Computing (HPC) cluster using Raspberry Pi’s. 
We created a multi-node cluster with a centralized storage server and integrated support for the SLURM workload manager.

In this post, I will take this HPC cluster a step further by detailing how to build a software stack. This will in principle enable users to easily and efficiently utilize packages and libraries on the system allowing for greater flexibility to meet specific computational needs.

### What is “software stack”?

In an HPC cluster, a software stack refers to the collection of software tools, libraries, compilers, and applications that are installed and configured to support various computing tasks. Setting up a software stack for an HPC cluster can be challenging but still essential for simplifying user experience, and ensuring flexibility and reproducibility.

There are two key aspects that I want to highlight when setting up a software stack for our HPC cluster:

#### 1. Computational Performance

Software stack in HPC systems is more than just providing users with convenient access to various packages and libraries; it also focuses on achieving high computational performance. In practice, HPC admins often create their compiler toolchain, a set of tools supporting libraries and header files, to help building packages from source to executable that can run efficiency on machine, as it offers numerous options for fine-tuning them to specific hardware configurations and utilizing latest available features. By carefully selecting and configuring the components of the stack, we can create a robust and efficient computational environment that maximizes the potential of an HPC cluster.

### 2. Shared storage

A centralized shared storage is also pivotal for software stack management across cluster nodes. It provides a single repository where software packages and configuration files are stored, ensuring consistency and simplifying updates. Shared storage integrates with environment modules enabling users to load specific software environments dynamically. High-speed network file systems ensure optimal performance for concurrent access from multiple nodes. I won’t delve into setting up a storage server with a parallel file system here. Instead, for simplicity, I’ll utilize our previously created NFS storage located at `/nfs/apps/`, or equivalently, a defined symbolic link to it as `/softwarestack`.

In what follows, I’ll guide you through the process of creating an example software stack, providing detailed steps and best practices to improve your SLURM HPC cluster.

### Environment module

An environment module is a software tool often used to dynamically modify the user environment. It provides a way to manage and customize the environment variables, paths, and settings required for different software applications and libraries. Environment modules are particularly useful in HPC clusters, where multiple versions of software and libraries are often required for various projects. In particular, Lua-based module (Lmod) is a powerful environment module system, designed using the Lua programming language, and is commonly used in HPC systems to manage software stack. It facilitates the efficient management of software environments by enabling users to dynamically load, unload, and switch between different versions of software packages.

Installing Lmod involves several steps to ensure it’s properly set up and integrated into your system. This will be our main building block for setting up the software stack in our cluster, so let’s proceed with the installation.

It is recommended to create separate admin account(s) to manage the software stack. 
However, for simplicity, I am using the “root” as the admin.

#### Lua dependency

Ensure that you have the necessary Lua dependency installed. 
You can install it via apt package manager or from the source using the following command

```bash
$ wget https://sourceforge.net/projects/lmod/files/lua-5.1.4.9.tar.bz2
$ tar xf lua-5.1.4.9.tar.bz2 && cd lua-5.1.4.9
$ ./configure --prefix=/softwarestack/lua/5.1.4.9
$ make && sudo make install
$ cd /softwarestack/lua && sudo ln -s 5.1.4.9 lua && sudo ln -s 5.1.4.9 luac
```

Here, we download the Lua source code, configure the installation with a specified path on shared software stack (`/softwarestack/lua/5.1.4.9`) , compile and install the source code. Finally, we create symbolic links (lua and luac) pointing to the versioned directory enabling easier access to the Lua interpreter and compiler.

Configure all nodes (e.g., `rpnode[01–02]`) to access the existing Lua installation on the shared storage.

```bash
$ sudo ln -s /softwarestack/lua/lua/bin/lua /usr/local/bin/
```

We are now able to easily utilize Lua from the shared storage on any node by creating this symbolic link.

#### Installing Lmod

Next, we install Lmod version 4.8 in `/softwarestack/lmod/4.8` using the following commands

```bash
$ sudo apt install tclsh 
$ wget https://sourceforge.net/projects/lmod/files/Lmod-8.4.tar.bz2 
$ tar xf Lmod-8.4.tar.bz2 && cd Lmod-8.4
$ ./configure --prefix=/softwarestack --with-fastTCLInterp=no 
$ sudo make install
```

We aim to set the default `MODULEPATH` to `/softwarestack/modulefiles`, where our custom module files will be defined later. 
It must be noted that a module file is a script in Lua that defines environment variables, paths, and settings required to use a specific software package or tool.

Let’s create the `modulefiles` directory and the default environment file, `StdEnv.lua` using the following commands:

```bash
$ sudo mkdir /softwarestack/modulefiles
$ sudo touch /softwarestack/modulefiles/StdEnv.lua
$ echo 'export MODULEPATH="/softwarestack/modulefiles"' | \
sudo tee -a  /softwarestack/lmod/lmod/init/profile /softwarestack/lmod/lmod/init/cshrc
```

Next, we will grant all nodes access (e.g., `rpnodes[01–02]`) to the Lmod initialization script for the bash and zsh shells, which are already included in the shared software stack.

```bash
$ sudo ln -s /softwarestack/lmod/lmod/init/profile /etc/profile.d/z00_lmod.sh 
$ sudo ln -s /softwarestack/lmod/lmod/init/cshrc   /etc/profile.d/z00_lmod.csh
```

After logging in again, we can verify that Lmod is working by using the module avail or module av commands.

<figure style="width: 800px" class="align-center">
  <img src="https://miro.medium.com/v2/resize:fit:750/format:webp/1*Tj6eThWM9URa7jgPRsdMdg.png" alt="">
  <figcaption>
  </figcaption>
</figure> 

At this stage, we do not have any modules added except for the default module `StdEnv`. 
However, I will add more modules by installing packages and defining module files accordingly in the following sections.

In the next three sections, I’ll explain adding of three modules to our software stack, all will be located on the shared storage. 
These modules are crucial for any HPC system: MPI library, Singularity container, and Conda package manager.

### MPI module

*Message Passing Interface* (MPI) is a critical component in HPC clusters, facilitating parallel computing and communication among nodes. MPI enables efficient utilization of the cluster’s resources in a large scale by allowing programs to be parallelized across multiple nodes. It must be noted that compiling MPI from source enables tailoring it to specific hardware configurations and integrating cutting-edge features that enhance performance and compatibility with specialized environments such as optimizations for Infiniband switches or CUDA-aware configuration for GPU computings. These capabilities are not typically available when installing MPI from standard binary distributions.
Building OpenMPI

OpenMPI is an open-source implementation of the MPI standard. Its high performance, portability, and robust features make it an invaluable tool for scientists, engineers, and researchers who need to perform large-scale computations efficiently.

Below, we build OpenMPI from source and make it available as the first module in our software stack as follows:

```bash
$ wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.0.tar.bz2$ tar xf openmpi-5.0.3.tar.bz2; cd openmpi-5.0.3
$ tar xf openmpi-4.1.0.tar.bz2 && cd openmpi-4.1.0
$ ./configure --prefix=/softwarestack/openmpi/4.1.0
$ sudo make install all
```

We need to create a module file within our `MODULEPATH`, located at `/softwarestack/modulefiles`. 
To better manage different versions of OpenMPI, we will place the file inside an OpenMPI directory and name it according to the specific version.

```bash
$ sudo mkdir /softwarestack/modulefiles/OpenMPI
$ sudo bash -c 'cat > /softwarestack/modulefiles/OpenMPI/4.1.0.lua << EOF
help([[
Description
===========
Open MPI Library

More information
================
 - Homepage: http://www.open-mpi.org/
]])
local base = "/softwarestack/openmpi/4.1.0/"
prepend_path("PATH", pathJoin(base, "bin"))
prepend_path("CPATH", pathJoin(base, "include"))
prepend_path("LD_LIBRARY_PATH", pathJoin(base, "lib"))
prepend_path("LIBRARY_PATH", pathJoin(base, "lib"))
conflict("OpenMPI")
EOF'
```

The lua script writes a help description, including a brief description and a homepage link, into the module file. 
It sets the base directory to `/softwarestack/openmpi/4.1.0/` and uses `prepend_path` to add the binary, include, and library directories to the respective environment paths (`PATH`, `CPATH`, `LD_LIBRARY_PATH`, and `LIBRARY_PATH`). 
Finally, it ensures no conflicting OpenMPI modules are loaded by using the conflict function. 
This configuration allows users to easily load the OpenMPI 4.1.0 environment using the module system.

Here’s how our module looks after adding the MPI module:

<figure style="width: 800px" class="align-center">
  <img src="https://miro.medium.com/v2/resize:fit:750/format:webp/1*ctUtVEZCKim_io9To7xvww.png" alt="">
  <figcaption>
  </figcaption>
</figure> 


Great! We can now easily load this module using the following command:

```bash
$ module load OpenMPI/4.1.0  # or, module load OpenMPI 
```

Let get more info about this module:

<figure style="width: 800px" class="align-center">
  <img src="https://miro.medium.com/v2/resize:fit:750/format:webp/1*7NgKH_L19m__8dY58uv0cg.png" alt="">
  <figcaption>
  </figcaption>
</figure> 


And _unload_ it with

```bash
$ module purge OpenMPI/4.1.0
```

Here are a few examples of how to use the MPI module with SLURM.

#### Example 1

Let’s create a simple MPI code in C and check if our OpenMPI module works. 
It will be used for testing and demonstrating MPI setup in a parallel computing environment.

```bash
$ bash -c 'cat > ~/mpi_hello_world.c << EOF
#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
  MPI_Init(NULL, NULL);

  // Get the number of processes
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Get the rank of the process
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  // Get the name of the processor
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);

  // Print off a hello world message
  printf("Hello world from processor %s, rank %d out of %d processors\n",
         processor_name, world_rank, world_size);

  // Finalize the MPI environment. No more MPI calls can be made after this
  MPI_Finalize();
}
EOF'
```

This command creates a C source file for an MPI “Hello World” program and saves it to `~/mpi_hello_world.c`. 
The program initializes the MPI environment, retrieves and prints the number of processes, the rank of each process, and the name of the processor it is running on.

Next, we compile this code using `mpicc`:

```bash
$ module load OpenMPI
$ mpicc mpi_hello_world.c -o helloworld.x
```

`mpicc` is a compiler wrapper provided by OpenMPI implementation specifically designed for compiling C programs that use the Message MPI. 
It simplifies the process of building MPI programs by automatically including the necessary compiler flags, library paths, and linker options required for MPI. 
The `mpi_hello_world.c` program prints a "hello world" message from all processors available.

We’re using the `salloc` command to allocate resources from SLURM dynamically. 
This allows us to request the necessary compute resources for our job interactively. 
The `--tasks` option specifies the number of parallel processes to run for the MPI code, ensuring that the appropriate number of tasks are allocated across the available nodes.

Output when running with `--tasks=2` processes:

<figure style="width: 800px" class="align-center">
  <img src="https://miro.medium.com/v2/resize:fit:750/format:webp/1*7NgKH_L19m__8dY58uv0cg.png" alt="">
  <figcaption>
  </figcaption>
</figure> 


And when we set `--tasks=4`

<figure style="width: 800px" class="align-center">
  <img src="https://miro.medium.com/v2/resize:fit:750/format:webp/1*MMRlMPrSXjgmMfq31xP87Q.png" alt="">
  <figcaption>
  </figcaption>
</figure> 


#### Example 2

The same example can be submitted as a batch job and for fun on multiple nodes using sbatch. For this, we require a batch job script as follows:

```bash
$ cat > ~/submit_mpi.sh << EOF
#!/bin/bash -l
#SBATCH --job-name=mpi               # Job name
#SBATCH --nodes=2                    # Number of nodes
#SBATCH --ntasks=4                   # Total number of MPI tasks
#SBATCH --ntasks-per-node=2          # Number of tasks per node
#SBATCH --time=00:05:00              # Time limit hrs:min:sec

# Load the MPI module
module load OpenMPI 

# Run the MPI program, "mpirun" is a synonym for "mpiexec"
mpirun -np $SLURM_NTASKS helloworld.x 

sleep 5
EOF
```

This SLURM batch script sets up and runs an MPI job named on 2 nodes, with a total of 4 MPI tasks distributed evenly across the nodes. 
It loads the OpenMPI module and executes the `helloworld.x` MPI program, ensuring the cluster's resources are utilized as specified.

After executing the job, you can monitor its status using the `squeue` command. 
Once completed, you can review the output to examine the results.

<figure style="width: 800px" class="align-center">
  <img src="https://miro.medium.com/v2/resize:fit:750/format:webp/1*_Ybbb0QYX0N9DKZqi_f6EA.png" alt="">
  <figcaption>
  </figcaption>
</figure> 

Please note that two processes are running on `rpnode01`, while the other two processes are running on `rpnode02`, nicely demonstrating MPI parallel processing across multiple nodes.

### Singularity module

Singularity is a containerization solution specifically designed for HPC clusters. 
It allows users to create and run containers that package applications along with their dependencies, ensuring consistent and reproducible environments. Singularity overcomes Docker’s limitations by running container processes without a daemon (it doesn’t touch cgroup). Also it effectively runs as the running user and doesn’t result in elevated access. We’ll install Singularity by following the instructions provided in the official documentation as outlined below.
Dependencies

We must first install development tools and libraries to be able to compile it from the source

```bash
sudo apt-get install -y \
    autoconf \
    automake \
    cryptsetup \
    fuse2fs \
    git \
    fuse \
    libfuse-dev \
    libglib2.0-dev \
    libseccomp-dev \
    libtool \
    pkg-config \
    runc \
    squashfs-tools \
    squashfs-tools-ng \
    uidmap \
    wget \
    zlib1g-dev
```

Singularity is written in GO, and may require a newer version of Go than is available in the repositories of the package distribution.

#### Installing GO

We’ll install Go inside the `/softwarestack` directory. We can also make this available as a environment module.

```bash
$ export VERSION=1.21.10 OS=linux ARCH=arm64

$ wget -O /tmp/go${VERSION}.${OS}-${ARCH}.tar.gz \
  https://dl.google.com/go/go${VERSION}.${OS}-${ARCH}.tar.gz

$ mkdir -p /softwarestack/go/${VERSION}
$ sudo tar -C /softwarestack/go/${VERSION} -xzf /tmp/go${VERSION}.${OS}-${ARCH}.tar.gz
```

This sequence first sets environment variables`VERSION`, `OS`, and `ARCH` to specify the GO version, target operating system, and architecture. 
It then downloads the GO binary distribution, creates a directory `/softwarestack/go/${VERSION}` to hold the installation, and extracts the downloaded archive into that directory.

Next, we create the module file for GO:

```bash
$ sudo mkdir /softwarestack/modulefiles/GO
$ sudo bash -c 'cat > /softwarestack/modulefiles/GO/${VERSION}.lua << EOF
help([[
Description
===========
GO Programming Language

More information
================
 - Homepage: https://go.dev 
]])
local base = "/softwarestack/go/${VERSION}/go"
prepend_path("PATH", pathJoin(base, "bin"))
conflict("GO")
EOF'
```

GO is now available in our software stack as module:

<figure style="width: 800px" class="align-center">
  <img src="https://miro.medium.com/v2/resize:fit:750/format:webp/1*r4lpYKM7q03ckMdkZ5cL6Q.png" alt="">
  <figcaption>
  </figcaption>
</figure> 


#### Building Singularity

We’ll use GO module to build Singularity with the following commands. First, download the desired version of Singularity:

```bash
$ export VERSION=4.1.3
$ git clone --recurse-submodules https://github.com/sylabs/singularity.git
$ cd singularity
$ git checkout --recurse-submodules v${VERSION}
```
Configure and build it from source (refer to this installation guide for details):

```bash
$ module load GO
$ ./mconfig --prefix=/softwarestack/singularity/${VERSION}
$ make -C builddir 
$ sudo make -C install builddir
```
and creating the module file:

```bash
$ sudo mkdir /softwarestack/modulefiles/Singularity
$ sudo bash -c 'cat > /softwarestack/modulefiles/Singularity/${VERSION}.lua << EOF
help([[
Description
===========
Singularity Container

More information
================
 - Homepage: https://sylabs.io/singularity 
]])
load("GO")
local base = "/softwarestack/singularity/${VERSION}"
prepend_path("PATH", pathJoin(base, "bin"))
conflict("Singulairty")
EOF'
```

This Lua script is used to create a module file for the Singularity container. 
The script loads the GO module as a dependency using load("GO"). It sets the base directory to `/softwarestack/singularity/${VERSION}`, where `${VERSION}` is a placeholder for the version of Singularity being configured. 
It adds the bin directory within base to the PATH environment variable using `prepend_path("PATH", pathJoin(base, "bin"))`. 
Finally, it ensures that this Singularity module does not conflict with any other Singularity modules that might be loaded simultaneously, using `conflict("Singularity")`.

With these steps completed, we now have a ready-to-use Singularity module integrated into our HPC software stack.

```bash
$ module load Singulairty
```

With both the GO and Singularity modules now loaded:

<figure style="width: 800px" class="align-center">
  <img src="https://miro.medium.com/v2/resize:fit:750/format:webp/1*X_sRPRuZF0kUeZW3DdoO8A.png" alt="">
  <figcaption>
  </figcaption>
</figure> 

We can proceed to explore examples that demonstrate how to utilize the Singularity container within SLURM.

#### Example 1

Let’s use the Singulairty module to pull and run a “hello world” Docker image.

```bash
$ module laod Singularity
$ singularity pull docker://arm64v8/hello-world 
$ srun singularity run hello-world_latest.sif
Hello from Docker!
This message shows that your installation appears to be working correctly.
...
```

First we load the Singularity module. Then, use `singularity pull` to download the Docker container image for the "hello-world" application. After downloading, run the container interactively with `srun singularity run hello-world_latest.sif`. This command executes the containerized "hello-world" application within the SLURM-managed environment.

#### Example 2

Alternatively, let’s build our PyTorch environment in a Singularity container using a torch.def file.

In Singularity, a definition file (often referred to as a `.def` file) is a text file used to define the contents and configuration of a Singularity container. This file typically includes instructions on how to build the container, specify its base operating system, install packages and dependencies, set environment variables, and configure other settings.

```bash
$ cat > ~/torch.def << EOF
Bootstrap: docker
From: ubuntu:20.04

%post
    apt-get -y update
    apt-get -y install python3-pip
    pip3 install numpy torch 

%environment
    export LC_ALL=C
EOF
```

Above `torch.def` file creates a Singularity container based on Ubuntu 20.04, installs Python 3 along with numpy and torch, and sets a locale environment variable. This setup is useful for users who need a reproducible environment for running Python applications that require these specific packages.

Let’s now build the container:

```bash
$ singularity build --fakeroot torch.sif torch.def
```

And finally run the Singularity container with SLURM `salloc` command

```bash
$ salloc --tasks=1 --cpus-per-task=2 --mem=1gb
$ srun singularity run torch.sif \
    python3 -c "import torch; print(torch.tensor(range(5)))"
$ exit
```

This command requests a resource allocation from SLURM, the workload manager. It specifies:

- `--tasks=1`: Allocate 1 task.
- `--cpus-per-task=2`: Allocate 2 CPUs per task.
- `--mem=1gb`: Allocate 1 GB of memory.

SLURM will allocate the requested resources and start an interactive session. 
Next, running the Python command within the Singularity container to verify the functionality of the PyTorch library, and then exit the interactive session, thereby releasing the resources.
Press enter or click to view image in full size


<figure style="width: 800px" class="align-center">
  <img src="https://miro.medium.com/v2/resize:fit:750/format:webp/1*Z4opbAG4DOkRBQnIBHUZqw.png" alt="">
  <figcaption>
  </figcaption>
</figure> 

Or, we can submit it as a batch job via `sbatch` command:

```bash
cat > ~/submit_torch.sh << EOF
#!/usr/bin/sh -l

#SBATCH --job-name=torch
#SBATCH --mem=1gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:05:00

module load Singularity

srun singularity run torch.sif \
    python3 -c "import torch; print(torch.tensor(range(5)))"
EOF
```

The screenshot below demonstrates how to submit the job, check the queue status, and view the output:

<figure style="width: 800px" class="align-center">
  <img src="https://miro.medium.com/v2/resize:fit:750/format:webp/1*IJQqNFE2H4jsmSdu8Z1PgA.png" alt="">
  <figcaption>
  </figcaption>
</figure> 


### Conda module

You’re likely familiar with Conda. 
It is an open-source package management and environment management system that allows users to install, run, and update packages beyond python in isolated environments. It is particularly popular in the data science and scientific computing communities due to its ability to manage multiple versions of software and libraries effortlessly.

Although users can install Conda directly in their home directory without needing admin access, I have found it particularly beneficial to provide Conda as a module with some preinstall environments. This approach offers hassle-free environments, especially for new users who may have less experience building packages from source, allowing them to use the default environment right out of the box.

Let’s download and install it using the following commands:


```bash
$ wget sudo wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
$ sudo bash Miniconda3-latest-Linux-aarch64.sh -b -u -p /softwarestack/miniforge
```

I used Miniforge instead of Miniconda as there is no official support from Anaconda for Raspberry Pi OS 64-bit on the Pi 4 at the moment.

Next, we need to create a module file to make Conda accessible to users.

```bash
$ sudo bash -c 'cat > /softwarestack/modulefiles/Conda.lua << EOF
help([[
Description
===========
Mini Forge Conda

More information
================
 - Homepage: https://github.com/conda-forge/miniforge 
]])
local base = "/softwarestack/miniforge"
prepend_path("PATH", pathJoin(base, "bin"))
conflict("Conda")
EOF'
```

You can see below that the Conda module is available.

<figure style="width: 800px" class="align-center">
  <img src="https://miro.medium.com/v2/resize:fit:750/format:webp/1*h_KUmS9hQwDtoof9K4EXbw.png" alt="">
  <figcaption>
  </figcaption>
</figure> 


#### Creating a default environment (admin)

Admin can create a default environment. For example, let’s create the `default-env` environment as follows:

```bash
$ sudo -i
$ module load Conda
$ conda update -n base conda
$ conda create default-env python
$ exit
```

After that, all users will be able to see this default environment.

<figure style="width: 800px" class="align-center">
  <img src="https://miro.medium.com/v2/resize:fit:750/format:webp/1*h_KUmS9hQwDtoof9K4EXbw.png" alt="">
  <figcaption>
  </figcaption>
</figure> 

HPC users do not have the privilege to install packages within the default environments. 
This restriction ensures that the default environments remain stable and consistent for all users, as any changes or updates to the packages within these environments can only be made by administrators.

#### Creating custom environment (user)

Regular users can create their own custom environments in their home directories under `~/.conda`. 
For example, as the user `pi`, you can create a custom environment as follows:

```bash
$ module load Conda
$ conda update -n base conda
$ conda create user-env python=3.10Your
```

User `pi` now has access to both the default environment and their custom-created environment.

<figure style="width: 800px" class="align-center">
  <img src="https://miro.medium.com/v2/resize:fit:750/format:webp/1*o5Nprp_HyGIqBruV5RI45A.png" alt="">
  <figcaption>
  </figcaption>
</figure> 

The user’s environment is visible only to that user and not to others. 
This makes it convenient to manage different types of Conda environments on the system. 
Lastly, the `pi` user is able to activate their custom environment.

<figure style="width: 800px" class="align-center">
  <img src="https://miro.medium.com/v2/resize:fit:750/format:webp/1*o5Nprp_HyGIqBruV5RI45A.png" alt="">
  <figcaption>
  </figcaption>
</figure> 


#### Example

This below example illustrates how to configure and run a job using a Conda environment managed by SLURM:

```bash
cat > ~/submit_conda.sh << EOF
#!/usr/bin/sh -l

#SBATCH --job-name=conda
#SBATCH --mem=1gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:05:00

# Load Conda and activate user-env
module load Conda
source activate user-env

srun python --version

sleep 5
EOF
```

This SLURM batch script allocates 1 GB of memory, requests 1 task with 2 CPUs, and sets a 5-minute time limit. 
The script loads the Conda module, activates the `user-env` environment, runs `srun python --version` to display the Python version within the activated environment.

<figure style="width: 800px" class="align-center">
  <img src="https://miro.medium.com/v2/resize:fit:750/format:webp/1*o5Nprp_HyGIqBruV5RI45A.png" alt="">
  <figcaption>
  </figcaption>
</figure> 


### Wrapping up

In this post, I’ve provided a few examples of how you can set up a software stack capable of running MPI applications, using the GO compiler, operating Singularity containers, and managing Conda environments with both default and custom configurations. In more complex scenarios, you may also need to build your compiler toolchain, such as GNU compilers or Intel compilers, tailored to your hardware, and compile everything with optimized flags specific to your hardware configuration.

If you want to learn about setting up an SLURM cluster, check out my earlier post about constructing an SLURM HPC cluster.
[Building a Slurm HPC Cluster with Raspberry Pi’s: Step-by-Step Guide](https://hghcomphys.github.io/slurm-hpc-cluster-with-raspberry-pis/)

Also feel free to check out my GitHub repository for a guide on setting up an HPC cluster that I created a few years ago. In this repository, I cover the setup process for the mentioned HPC features. Keep in mind that the information is somewhat outdated and might require some adjustments to work with current versions.

<a href="https://github.com/hghcomphys/raspi-hpc-cluster">
  <i class="fab fa-github"></i> Raspi-HPC-Cluster GitHub Repository
</a>

<!-- GitHub - hghcomphys/raspi-hpc-cluster: A repo for setting up a test but scalable HPC cluster using… -->
<!-- A repo for setting up a test but scalable HPC cluster using Raspberry Pi. - hghcomphys/raspi-hpc-cluster.github.com -->

I hope you find this post helpful for setting up your HPC software stack.

Thank you for reading!
