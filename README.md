# README.md

## These are instructions to compile and run the code

This project primarily utilized docker images from the GNU Compiler Collection and NVIDIA to make development with different environments easy, make cross-compatible, reproducible results and make installation easy.

### Get started with the CPU-implementation

This one is easy, you can do it on anything that'll install docker, but I'd recommend using WSL if you're on Windows:

#### Step 1: Install [docker](https://www.docker.com/)

#### Step 2: Clone the repo

```bash
git clone https://github.com/ostew5/kmer-signatures.git
```

#### Step 3: Run the run file

```bash
cd kmer-signatures
sh run_cpu
```

### Get started with the GPU-implementation

This is surprisingly easy, I promise, you've got this. You'll need a NVIDIA GPU that supports CUDA version 13.

#### Step 0: Enable WSL 2 if you're using Windows

`nvidia-smi` needs WSL 2, if it isn't activated open PowerShell as Admin and run:

```powershell
wsl --install
```

#### Step 1: Install [docker](https://www.docker.com/)

#### Step 2: Install [NVIDIA container tools](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

#### Step 3: Clone the repo

```bash
git clone https://github.com/ostew5/kmer-signatures.git
```

#### Step 4: Run the run file

```bash
cd kmer-signatures
sh run_cuda
```

#### Step 5: Optionally run the profiling script

This uses NVIDIA Nsight Compute, which needs a GPU with Compute Capability 5.0 or newer

```bash
sh run_cuda-profiling
```
