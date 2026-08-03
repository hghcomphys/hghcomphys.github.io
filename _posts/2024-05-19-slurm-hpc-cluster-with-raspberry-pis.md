---
title: "Building a Slurm HPC Cluster with Raspberry Pi’s: Step-by-Step Guide"
categories:
  - Python 
tags:
  - HPC 
  - SLURM
  - Raspberry Pi
header:
  image: &image "https://miro.medium.com/v2/resize:fit:720/format:webp/1*2VhYAZPXdUHuA6ReJ1BfJg.jpeg"
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


In this post, I’ll share my attempt to build a _Slurm_ High-Performance Computing (HPC) cluster using Raspberry Pi’s. 
I started off using this cluster a while ago as a test to create a bigger HPC cluster that also supports GPU computing. 
I got hands-on experience with various components of an HPC setup and figured out how they all fit together. 
Setting up SLURM components was indeed a main part of it, and after some examination, I finally got my own HPC cluster up and running. 
Since setting up this machine is pretty straightforward, it’s been great for quickly trying out different software packages and libraries, or tweaking the cluster’s hardware to see what works best.

The goal here is to build an HPC cluster that can handle multiple compute nodes. 
Creating such systems from scratch is a challenging endeavor that requires some expertise. 
Additionally, locating a comprehensive tutorial that covers all the necessary steps for configuring a Slurm cluster can be quite difficult, at least based on my experience. 
Regarding that, I hope you find this step-by-step guide useful on your journey to build your HPC cluster.


### Contents

I’ve structured this post as it begins with a short introduction to HPC and the significance of Slurm as a commonly used resource manager and job scheduler for HPC systems. Following that, I’ll demonstrate the cluster network topology, discussing the essential prerequisites, hardware specifications, and operating system setup. Afterward, I’ll guide you through setting up the storage node. Then, through building Slurm from source, installing, and configuring it on the master node, along with the addition of an extra compute node. Lastly, I’ll present a few examples showcasing the status of our constructed Slurm cluster and how job submissions are handled within it.
Features

In the end, you’ll be all set to have your own HPC cluster utilizing

1. Slurm workload manager
2. Centralized network storage

For now, I’ve kept my focus on minimal features here to prevent this post from becoming too lengthy. The aim is to first set up an HPC cluster with essential functionality, yet flexible enough to be expanded upon later.

### What is an HPC cluster?

An HPC cluster is a network of interconnected computers, designed to collectively solve computational problems and process large datasets at high speeds across numerous fields. These clusters consist of multiple compute nodes, each equipped with processors, memory, and often specialized accelerators like GPUs which enable researchers and scientists to tackle computationally demanding tasks and simulations. Slurm, which stands for Simple Linux Utility for Resource Management, is an open-source HPC job scheduler and resource manager. It plays a crucial role in efficiently allocating computing resources, managing job scheduling, and orchestrating parallel computations on HPC clusters. As an alternative to Kubernetes clusters, Slurm specializes in managing batch computing workloads typically found in scientific research, simulations, and data analysis tasks. While Kubernetes focuses more on containerized applications and microservices.
Cluster network topology

There are various network topologies available for configuring an HPC cluster, each tailored to specific performance expectations and application requirements. In our scenario, we’re focusing on interconnecting three Raspberry Pi’s within a subnet (`10.0.0.x`) via a router and a switch. I’ve set up a DHCP server on the router with reserved IPs to allocate fixed IP addresses to each Raspberry Pi based on their unique MAC address. However, this can be simplified by directly configuring static IPs (`192.168.0.x`) on the Raspberry Pi’s themselves. And, if you’re using a Wi-Fi access point instead of a LAN connection to interconnect Raspberry Pi’s through your home ISP router, you can bypass the need for the router and ethernet switch. Since this setup is intended for a test cluster and doesn’t require a fast or even stable connection, this alternative is viable. Furthermore, internet connectivity is provided by linking the router to the home ISP Wi-Fi access point and sharing it to devices via ethernet. The diagram below provides a visual representation of the cluster network:

<figure style="width: 800px" class="align-center">
  <img src="https://miro.medium.com/v2/resize:fit:720/format:webp/1*ainKot9AHAjvo4pZXYR-Yw.png" alt="">
  <figcaption>
Raspberry Pi HPC Cluster network topology
  </figcaption>
</figure> 


This network topology offers the advantage of portability, allowing the cluster to be easily connected to different access points. Additionally, I’ve used one node solely for data management, serving as a dedicated network storage server. The remaining two Raspberry Pi’s are utilized one as the master node and as well as a computing node, while the other one serves as an additional compute node.

### Prerequisite

#### Hardware components

I used 3x Raspberry Pi’s that had been sitting idle for some time and some other components as the following:

1. Raspberry Pi 4 Model B 2GB board (hostname `rpnode01`).
   This device serves as the master node and compute node.
2. Raspberry Pi 4 Model B 2GB board (hostname `rpnode02`).
   This operates as a second compute node.
3. Raspberry Pi 3+ 1GB board (hostname `filenode01`).
   It is set up as a network storage server.
4. USB power hub.
   You’ll need a USB power charger with multiple ports capable of supporting the power requirements of multiple Raspberry Pi’s simultaneously (at least 2A for each Raspberry Pi).
5. A router and Ethernet switch (optional).
   The router will manage external connections and the switch handles internal device communications.


<figure style="width: 600px" class="align-center">
  <img src="https://miro.medium.com/v2/resize:fit:720/format:webp/1*R0C_j6zz34QpMZfC0cA6zQ.jpeg" alt="">
  <figcaption>
A photo of the Raspberry Pi cluster
  </figcaption>
</figure> 


#### Operating system

I used *Debian bookworm* (version 12) OS Lite 64-bit is known for being lightweight, which can be beneficial for systems with limited resources like Raspberry Pi’s. The following adjustments are expected to be applied to all devices.

1. I used the default user `pi` with given dummy password of `testpass`. 
But, there’s room for improvement here by utilizing tools such as LDAP or other authentication mechanisms to keep user and group IDs synced across the cluster.

2. Enable SSH access on nodes. 
   A convenient way would be using SSH key sharing to access the nodes.
    Enable control group for CPU and memory. 
    Modify `/boot/firmware/cmdline.txt` by adding 
    ```text
    cgroup_enable=cpuset cgroup_memory=1 cgroup_enable=memory swapaccount=1
    ```
    Reboot the system after this change.
3. Add the below hostnames to `/etc/hosts`
   ```bash
    10.0.0.1 rpnode01 rpnode01.home.local
    10.0.0.2 rpnode02 rpnode02.home.local
    10.0.0.3 filenode01 filenode01.home.local
    ```
4. Configure language and regional settings using (if needed)
   ```bash
    $ sudo raspi-config
    ```
5. Finally, update and upgrade system packages
   ```bash
    $ sudo apt update && sudo apt upgrade
    ```

<figure style="width: 600px" class="align-center">
  <img src="https://miro.medium.com/v2/resize:fit:720/format:webp/1*ui8V0o_JDLxQpMkKe5szpg.png" alt="">
  <figcaption>
  </figcaption>
</figure> 


#### Storage node

In an HPC cluster, compute nodes are designed to be stateless, meaning they do not retain any persistent data or state. Instead, all application software and user data are stored on a centralized shared storage. This architecture offers several advantages. First, it enhances scalability by simplifying the addition of new compute nodes without the need to replicate data across multiple machines. Second, it provides flexibility to users, allowing them to access their applications and data from any compute node in the cluster. Third, storing all data on a centralized storage node ensures data integrity and consistency, eliminating concerns about inconsistencies that may arise from storing data locally on individual compute nodes. Lastly, the stateless compute node architecture simplifies maintenance tasks such as software updates, hardware replacements, and troubleshooting, as there is no need to transfer or backup data stored locally on compute nodes.


##### NFS server

I set up an NFS (Network File System) server on the dedicated node `filenode01` where we can have network storage. When I refer to a compute node as stateless in the context of an HPC cluster, I mean that the compute node itself does not retain any persistent data. Instead, all user home directory data and application software reside on a centralized storage node.

For that, I installed the NFS server the apt package manager. This can be done by running the following command in the terminal:

```bash
$ sudo apt install nfs-kernel-server
```

Then I defined the directories I wanted to share over NFS by configuring the `/etc/exportsfile`. It is a configuration file used by the NFS server in Unix-like operating systems. This file specifies which directories on the server are shared with NFS clients and defines the access permissions for those directories. I ensured those folders existed and then added entries using the following commands:

```bash
$ sudo mkdir -p /home /nfs
```

```bash
$ sudo bash -c "cat >> /etc/exports << EOF
/home  *(rw,sync,no_root_squash,no_subtree_check)
/nfs  *(rw,sync,no_root_squash,no_subtree_check)
EOF"
```
Here I used `*` to allow access from any nodes and specified options as needed, such as `rw` for read-write access. 
After editing the exports file, apply the changes by running:

```bash
$ sudo exportfs -ra
```

If you have a firewall enabled on the node, you may need to open the NFS ports. 
NFSv4 uses TCP and UDP ports 2049, while NFSv3 uses additional ports. 
You can open these ports using **ufw** or **iptables**, depending on your firewall configuration.

We can verify that the NFS share is available by `showmount`. 
This command will display the list of exported directories on the node.

```bash
$ sudo showmount -e
Export list for filenode01:
/home *
/nfs *
```

##### NFS Client

To enable network storage access on both NFS client nodes, namely `rpnode01` and `rpnode02`, we can do this by adjusting the `/etc/fstab` file to incorporate the NFS mount points. This is a system file that allows automating the mounting filesystems during the system startup.

Make sure again that the directories `/home` and `/nfs` exist on the client side before making changes to this file. Execute the following command:

```bash
$ sudo mkdir -p /home /nfs

$ sudo bash -c "cat >> /etc/fstab << EOF
filenode01:/home /home nfs defaults 0 0
filenode01:/nfs /nfs nfs defaults 0 0
EOF"
```

Every line appended to the `/etc/fstab` defines a distinct NFS mount point. 
This instructs the system to mount the `/home` (`/nfs`) directory from the NFS server with the `hostname filenode01` to the local `/home` (`/nfs`) directory. As it’s already mentioned, the `/home` is designated for users’ data, and the `/nfs` directory is intended for a shared software stack. Reboot the node.

To check the correctness of the updated `/etc/fstab` I use the below command mount `-a`.

```bash
$ sudo mount -av
/nfs    : successfully mounted
/home    : successfully mounted
```

That reads `/etc/fstab` as the system starts, it mounts the filesytems that are not yet mounted.


#### Master node

##### Building and installing Slurm

I always prefer to compile Slurm from its source code rather than using pre-built packages. This allows for customization, ensures access to the latest features and fixes, and provides educational value. Instructions for building the Slurm package from the source and making it ready-to-install are pretty much as stated here except I applied a few changes which I’ll explain in what follows.

###### 1. Prerequisite libraries

Before proceeding with the configuration of Slurm, ensure that the following libraries or header files are installed. You can easily install them via apt package manager.

```bash
$ sudo apt install libpmix-dev libpam-dev libmariadb-dev \
  libmunge-dev libdbus-1-dev munge
```

###### 2. Making from source

Let’s download the most recent version of Slurm version 23.11, as I’m writing this post, from SchedMD Github repository. 
We’re going to build it for *aarch64* architecture instead of x86_64.

```bash
$ sudo mkdir /opt/slurm && cd /opt/slurm
$ sudo wget https://github.com/SchedMD/slurm/archive/refs/tags/slurm-23-11-6-1.tar.gz
$ sudo tar -xf slurm-23-11-6-1.tar.gz
```
Compiling Slurm will require several minutes, considering that we’re making everything from scratch.

```bash
$ cd slurm-slurm-23-11-6-1
$ sudo ./configure \
  --prefix=/opt/slurm/build \
  --sysconfdir=/etc/slurm \
  --enable-pam \
  --with-pam_dir=/lib/aarch64-linux-gnu/security/ \
  --without-shared-libslurm \
  --with-pmix 
$ sudo make 
$ sudo make contrib
$ sudo make install
```
The `--prefix` option specifies the base directory for installing the compiled code. Instead of installing directly into `/usr`, I’ve set it to `/opt/slurm/build`. The reason is the intention to utilize it for creating an installable package when installing Slurm on additional compute nodes.

###### 3. Building the Debian package

I used the `fpmtool` for the creation of a Debian package of the compiled codes. This requires installing an additional package.

```bash
$ sudo apt ruby-dev 
$ sudo gem install fpm
```

This tool will create a package `fileslurm-23.11_1.0_arm64.deb`. 
It’s worth mentioning that beginning with Slurm 23.11.0, Slurm includes the files required to build Debian packages.

```bash
$ sudo fpm -s dir -t deb -v 1.0 -n slurm-23.11 --prefix=/usr -C /opt/slurm/build .
Created package {:path=>"slurm-23.11_1.0_arm64.deb"}
```

###### 4. Installing Debian package

Next, we install this package via `dpkg` command:

```bash
$ sudo dpkg -i slurm-23.11_1.0_arm64.deb
Preparing to unpack slurm-23.11_1.0_arm64.deb ...
Unpacking slurm-23.11 (1.0) over (1.0) ...
Setting up slurm-23.11 (1.0) ...
Processing triggers for man-db (2.11.2-2) ...
```

We must also create the `slurm` system user and initialize the required directories with correct access permissions. 
Make sure the `slurm` user exists and its user ID is synchronized across the cluster. 
Note that files and directories used by Slurm controller will need to be readable or writable by the `slurm` user. 
In addition, the log file directory `/var/log/slurm` and state save directory `var/spool/slurm` must be writable.

Here, I fixed the `slurm` user and group IDs to 151.

```bash
$ sudo adduser --system --group -uid 151 slurm
```

Also creating the necessary directories with expected permission via executing the following commands:

```bash
$ sudo mkdir -p /etc/slurm /var/spool/slurm/ctld /var/spool/slurm/d /var/log/slurm
$ sudo chown slurm: /var/spool/slurm/ctld /var/spool/slurm/d /var/log/slurm
```

##### Slurm configuration

So far, everything is progressing well. 
Up to this point, we have built and installed the Slurm packages. 
The next step will be to configure its various components and run them as services.

###### 1. Slurm database daemon

We’re going to set up the Slurm database daemon (`slurmdbd`) to gather detailed accounting information for each job, with all accounting data being stored in a database. 
This necessitates first the creation of a database server on the master node, but ideally, it should be on a separate node. 
I’ve opted for MariaDB, an open-source database that is compatible with MySQL. You can deploy the database server with the following instructions


```bash
$ sudo apt install mariadb-server
$ sudo mysql -u root
create database slurm_acct_db;
create user 'slurm'@'localhost';
set password for 'slurm'@'localhost' = password('slurmdbpass');
grant usage on *.* to 'slurm'@'localhost';
grant all privileges on slurm_acct_db.* to 'slurm'@'localhost';
flush privileges;
exit
```

After that, we need to create `/etc/slurm/slurmdbd.conf` and add the required configurations such as specifying authentication, database server hostname, logging, etc. Execute the below command to add the configuration file.

```bash
$ sudo bash -c "cat > /etc/slurm/slurmdbd.conf << EOF
# Authentication info
AuthType=auth/munge

# slurmDBD info
DbdAddr=localhost
DbdHost=localhost
SlurmUser=slurm
DebugLevel=3
LogFile=/var/log/slurm/slurmdbd.log
PidFile=/run/slurmdbd.pid
PluginDir=/usr/lib/slurm

# Database info
StorageType=accounting_storage/mysql
StorageUser=slurm
StoragePass=slurmdbpass
StorageLoc=slurm_acct_db
EOF"
```
This file describes the Slurm database daemon configuration information. 
Please note that it should be only on the computer where `slurmdbd` executes. 
Also, it must be readable only by the `slurm` user.

```bash
$ sudo chmod 600 /etc/slurm/slurmdbd.conf
$ sudo chown slurm: /etc/slurm/slurmdbd.conf
```

Next, we need to set up `slurmdbd` as a `systemd` service. This can be done by creating `/ets/system/systemd/slurmdbd.service`.

```bash
$ sudo bash -c "cat > /etc/systemd/system/slurmdbd.service << EOF
[Unit]
Description=Slurm DBD accounting daemon
After=network.target munge.service
ConditionPathExists=/etc/slurm/slurmdbd.conf

[Service]
Type=forking
EnvironmentFile=-/etc/sysconfig/slurmdbd
ExecStart=/usr/sbin/slurmdbd $SLURMDBD_OPTIONS
ExecReload=/bin/kill -HUP $MAINPID
PIDFile=/run/slurmdbd.pid

[Install]
WantedBy=multi-user.target
EOF"
```

You can now enable and start `slurmdbd.service` as follows:

```bash
$ sudo systemctl enable slurmdbd.service
$ sudo systemctl start slurmdbd.service

$ sudo systemctl | grep slurmdbd
  slurmdbd.service      loaded active running   Slurm DBD accounting daemon
```

If everything goes well, you should see the `slurmdbd` service up and running. 
Otherwise, please check `/var/log/slurm/slurmdbd.log` file or the `systemd status`.

###### 2. Slurm controller daemon

Slurm Controller Daemon (`slurmctl`) orchestrates Slurm activities and it is the central management daemon of Slurm. 
It monitors all other Slurm daemons and resources, accepts jobs, and allocates resources to those jobs. 
We must create a `/etc/slurm/slurmd.conf` file. 
This configuration file defines how Slurm interacts with resources, manages jobs, and communicates with other components. 
It includes a wide variety of parameters and it must be available on each node of the cluster consistently. 
Use the following command in the terminal to create and add the required configurations

```bash
$ sudo bash -c "cat > /etc/slurm/slurm.conf << EOF
ClusterName=raspi-hpc-cluster
ControlMachine=rpnode01
SlurmUser=slurm
AuthType=auth/munge
StateSaveLocation=/var/spool/slurm/ctld
SlurmdSpoolDir=/var/spool/slurm/d
SwitchType=switch/none
MpiDefault=pmi2
SlurmctldPidFile=/run/slurmctld.pid
SlurmdPidFile=/run/slurmd.pid
ProctrackType=proctrack/cgroup
PluginDir=/usr/lib/slurm
ReturnToService=1
TaskPlugin=task/cgroup

# SCHEDULING
SchedulerType=sched/backfill
SelectTypeParameters=CR_Core_Memory,CR_CORE_DEFAULT_DIST_BLOCK,CR_ONE_TASK_PER_CORE

# LOGGING
SlurmctldDebug=3
SlurmctldLogFile=/var/log/slurm/slurmctld.log
SlurmdDebug=3
SlurmdLogFile=/var/log/slurm/slurmd.log
JobCompType=jobcomp/none

# ACCOUNTING
JobAcctGatherType=jobacct_gather/cgroup
AccountingStorageTRES=gres/gpu
DebugFlags=CPU_Bind,gres
AccountingStorageType=accounting_storage/slurmdbd
AccountingStorageHost=localhost
AccountingStoragePass=/run/munge/munge.socket.2
AccountingStorageUser=slurm
AccountingStorageEnforce=limits

# COMPUTE NODES
NodeName=rpnode01 CPUs=4 Sockets=1 CoresPerSocket=4 ThreadsPerCore=1 RealMemory=1800 State=idle
NodeName=rpnode02 CPUs=4 Sockets=1 CoresPerSocket=4 ThreadsPerCore=1 RealMemory=1800 State=idle

# PARTITIONNS
PartitionName=batch Nodes=rpnode[01-02] Default=YES State=UP DefaultTime=1-00:00:00 DefMemPerCPU=200 MaxTime=30-00:00:00 DefCpuPerGPU=1
EOF"
```

And again creating `/etc/system/systemd/slurmctld.service` to run `slurmctl` as a systemd daemon. 
This can be added by running the following command

```bash
$ sudo bash -c "cat > /etc/systemd/system/slurmctld.service << EOF
[Unit]
Description=Slurm controller daemon
After=network.target munge.service
ConditionPathExists=/etc/slurm/slurm.conf

[Service]
Type=forking
EnvironmentFile=-/etc/sysconfig/slurmctld
ExecStart=/usr/sbin/slurmctld $SLURMCTLD_OPTIONS
ExecReload=/bin/kill -HUP $MAINPID
PIDFile=/run/slurmctld.pid

[Install]
WantedBy=multi-user.target
EOF"
```

We’re now ready to enable and start `slurmctld.service` as follows:

```bash
$ sudo systemctl enable slurmctld.service
$ sudo systemctl start slurmctld.service

$ sudo systemctl | grep slurmctld
slurmctld.service       loaded active running   Slurm controller daemon
```

##### 3. Slurm node daemon

If you want to use the master node also as a compute node then you should set up the compute node daemon of Slurm (`slurmd`). 
The `slurmd` daemon must be executed on every compute node. 
It monitors all tasks running on the compute node, accepts jobs, launches tasks, and kills running tasks upon request. 
This daemon reads `slurmd.conf` together with two additional files: `cgroup.conf` and `cgroup_allowd_devices_file.conf`. 
Use the following commands to create the two required control group (cgroup) files as follows:

```bash
$ sudo bash -c "cat > /etc/slurm/cgroup.conf << EOF
ConstrainCores=yes 
ConstrainDevices=yes
ConstrainRAMSpace=yes
EOF"
```

```bash
$ sudo bash -c "cat > /etc/slurm/cgroup_allowed_devices_file.conf << EOF
/dev/null
/dev/urandom
/dev/zero
/dev/sda*
/dev/cpu/*/*
/dev/pts/*
/dev/nvidia*
EOF"
```

Then, we must once again create the `systemd.service` file to run `slurmd` as a service.

```bash
$ sudo bash -c "cat > /etc/systemd/system/slurmd.service << EOF
[Unit]
Description=Slurm node daemon
After=network.target munge.service
ConditionPathExists=/etc/slurm/slurm.conf

[Service]
Type=forking
EnvironmentFile=-/etc/sysconfig/slurmd
ExecStart=/usr/sbin/slurmd -d /usr/sbin/slurmstepd $SLURMD_OPTIONS
ExecReload=/bin/kill -HUP $MAINPID
PIDFile=/run/slurmd.pid
KillMode=process
LimitNOFILE=51200
LimitMEMLOCK=infinity
LimitSTACK=infinity
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=multi-user.target
EOF"
```

Finally enabling and starting `slurmd` service using the below commands:

```bash
$ sudo systemctl enable slurmd.service
$ sudo systemctl start slurmd.service

$ sudo systemctl | grep slurmd
  slurmd.service       loaded active running   Slurm node daemon
```

At this step, we’re all set and we can view information about our Slurm nodes and partitions using Slurm `sinfo` command:

```bash
$ sinfo
PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST
batch*       up 30-00:00:0      1   idle rpnode01
```

If you see this output, it means that you have successfully installed and configured Slurm. 

Well done!


#### Slurm Accounting

Slurm collects accounting information for every job and job step executed. 
It also supports accounting records being directly to the database. 
For testing purposes, we define a cluster “raspi-hpc-cluster” and an account “compute” in the Slurm database as follows

```bash
$ sudo sacctmgr add cluster raspi-hpc-cluster
$ sudo sacctmgr add account compute description="Compute account" Organization=home

$ sudo sacctmgr show account
   Account                Descr                  Org 
---------- -------------------- -------------------- 
   compute      Compute account                home 
      root default root account                root
```

And associating user pi as a regular Slurm account

```bash
$ sudo sacctmgr add user pi account=compute
$ sudo sacctmgr modify user pi set GrpTRES=cpu=4,mem=1gb
$ sudo sacctmgr show user
      User   Def Acct     Admin 
---------- ---------- --------- 
        pi    compute      None 
      root       root Administ+
```

If everything goes well, our single-node Slurm cluster must now be ready for a job submission. 
Let’s first display information about the current state of the available node

```bash
$ sinfo 
PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST
batch*       up 30-00:00:0      1   idle rpnode[01]
```

Now, let’s execute a simple Slurm srun command and check the output.

```bash
$ srun hostname
rpnode01
```

This indicates that our job ran successfully as a Slurm job and returned the hostname of the compute node, which in this case is `rpnode01`.

#### Compute nodes

Expanding our Slurm cluster to incorporate additional compute nodes involves several key steps as follows:

1. Install prerequisite libraries and header files.
2. Copy `/etc/munge/munge.key`, from the master node to the compute node, change the owner to `munge` user, and restart `munge.service`
3. Install the `slurm-23.11_1.0_arm64.deb`
4. Create `slurm` user and required Slurm directories.
5. Copy `slurm.conf`, `cgroup.conf` , and `cgroup_allowed_devices_file.conf` files to `/etc/slurm/`
6. Enable and start `slurmd.service`

#### Testing

Let’s first update the state of the new `rpnode02` node to idle using

```bash
$ scontrol update nodename=rpnode02 state=idle

$ sinfo 
PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST
batch*       up 30-00:00:0      2   idle rpnode[01-02]
```

And again run the `hostname` job on the new node using

```bash
$ srun -w rpnode02 hostname
rpnode02
```

As you can see, this job was executed on the second compute node, so it returns the `rpnode02` this time.

### Examples

#### Cluster information

Partitions in Slurm are a way to divide a cluster into logical sets of nodes that can be used to manage and allocate resources more effectively. 
Here, we display detailed information about the batch partition configured in the Slurm

```text
$ scontrol show partition
PartitionName=batch
   AllowGroups=ALL AllowAccounts=ALL AllowQos=ALL
   AllocNodes=ALL Default=YES QoS=N/A
   DefaultTime=1-00:00:00 DisableRootJobs=NO ExclusiveUser=NO GraceTime=0 Hidden=NO
   MaxNodes=UNLIMITED MaxTime=30-00:00:00 MinNodes=0 LLN=NO MaxCPUsPerNode=UNLIMITED MaxCPUsPerSocket=UNLIMITED
   Nodes=rpnode[01-02]
   PriorityJobFactor=1 PriorityTier=1 RootOnly=NO ReqResv=NO OverSubscribe=NO
   OverTimeLimit=NONE PreemptMode=OFF
   State=UP TotalCPUs=8 TotalNodes=2 SelectTypeParameters=NONE
   JobDefaults=DefCpuPerGPU=1
   DefMemPerCPU=200 MaxMemPerNode=UNLIMITED
   TRES=cpu=8,mem=3600M,node=2,billing=8
```

And showing the `rpndoe01` status

```text
$ scontrol show nodes
NodeName=rpnode01 Arch=aarch64 CoresPerSocket=4 
   CPUAlloc=0 CPUEfctv=4 CPUTot=4 CPULoad=0.01
   AvailableFeatures=(null)
   ActiveFeatures=(null)
   Gres=(null)
   NodeAddr=rpnode01 NodeHostName=rpnode01 Version=23.11.6
   OS=Linux 6.6.28+rpt-rpi-v8 #1 SMP PREEMPT Debian 1:6.6.28-1+rpt1 (2024-04-22) 
   RealMemory=1800 AllocMem=0 FreeMem=297 Sockets=1 Boards=1
   State=IDLE ThreadsPerCore=1 TmpDisk=0 Weight=1 Owner=N/A MCS_label=N/A
   Partitions=batch 
   BootTime=2024-05-19T13:58:15 SlurmdStartTime=2024-05-19T14:20:03
   LastBusyTime=2024-05-19T14:26:11 ResumeAfterTime=None
   CfgTRES=cpu=4,mem=1800M,billing=4
   AllocTRES=
   CapWatts=n/a
   CurrentWatts=0 AveWatts=0
   ExtSensorsJoules=n/a ExtSensorsWatts=0 ExtSensorsTemp=n/a
```

Submitting jobs

Let’s create a simple Slurm batch file in the home directory

```bash
$ cat > ~/submit.sh << EOF
#!/usr/bin/sh

#SBATCH --job-name=testjob
#SBATCH --mem=10mb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:01:00

srun sleep 10
EOF
```
Now we can submit this job

```bash
$ sbatch submit.sh 
Submitted batch job 8
```

You can see the status of the submitted job in the default batch queue

```bash
$ squeue -al
Sun May 19 14:26:03 2024
JOBID PARTITION     NAME     USER    STATE       TIME TIME_LIMI  NODES NODELIST(REASON)
    8     batch  testjob       pi  RUNNING       0:03      1:00      1 rpnode01
```

This job is running on `rpnode01` and with a time limit of 1 minute.


### Wrapping up

As I mentioned earlier, this HPC cluster serves as a test environment. It is currently equipped with fundamental Slurm and central storage functionalities but it has potential for future expansion and enhancement. I have plans to write subsequent posts covering other topics related to HPC system setup including user accounting, disk quota, setting up software stacks utilizing environment modules and Conda package manager, MPI implementation, and setting up a Jupyterhub service.

Feel free to check out my GitHub repository for a guide on setting up an HPC cluster that I created a few years ago. 

<a href="https://github.com/hghcomphys/raspi-hpc-cluster">
  <i class="fab fa-github"></i> Raspi-HPC-Cluster GitHub Repository
</a>

In this repository, I cover the setup process for the mentioned HPC features. 
Keep in mind that the information is somewhat outdated and might require some adjustments to work with current versions.

<!-- GitHub - hghcomphys/raspi-hpc-cluster: A repo for setting up a test but scalable HPC cluster using… -->
<!-- A repo for setting up a test but scalable HPC cluster using Raspberry Pi. - hghcomphys/raspi-hpc-cluster.github.com -->

Hopefully, this post has been a good starting point. 

Thanks for reading!
