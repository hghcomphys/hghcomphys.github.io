---
title: "Setup a Raspberry-Pi HPC Cluster"
categories:
  - High-performance computing
tags:
  - Slurm
  - Raspberry-pi
  - Jupyterhub
header:
  image: &image "https://github.com/hghcomphys/raspi-hpc-cluster/blob/master/docs/raspi_cluster.JPG?raw=true"
  caption: "Raspberry-Pi Nodes"
  teaser: *image
# link: https://github.com/hghcomphys/raspi-hpc-cluster
# classes: wide
# toc: true
# toc_label: "Table of Contents"
# toc_icon: "cog"
# layout: splash
# classes:
#   - landing
#   - dark-theme
---

In this post, I explain steps on how to setup a __test__ but __scalable__  high-performance computing (HPC) cluster using [Raspberry Pi](https://en.wikipedia.org/wiki/Raspberry_Pi) and with a focus on data science. The experiences learned from this tutorial are intended to hopefully help you to build your real HPC cluster, e.g. with hundreds of compute nodes, which support both interactive and command-line interface.

### Feature implemented
- [Slurm][slurmref] workload manager
- Batch job submission
- [Jupyterhub](https://jupyter.org/hub) service integrated to Slurm
- Network file share (NFS)
- User and group disk quota
- Conda package management for Python/R
- Environment module management using [Lmod](https://lmod.readthedocs.io/en/latest/)
- Support parallel [MPI](https://www.open-mpi.org/) applications integrated with Slurm


**Note:** Based on my experience, setting up a real HPC cluster is very similar to what we are doing here using Raspberry Pi. 
{: .notice--info}

**Info:** Generic resource scheduling such as Graphics Processing Units (GPUs) and Intel Many Integrated Core (MIC) processors are supported by [Slurm][slurmref] and they can be easily added later through a flexible plugin mechanism on real HPC systems. 
{: .notice--info}

For a full description and required configuration files please see 
[https://github.com/hghcomphys/raspi-hpc-cluster](https://github.com/hghcomphys/raspi-hpc-cluster)


<!-- References -->
[slurmref]: https://slurm.schedmd.com/overview.html
