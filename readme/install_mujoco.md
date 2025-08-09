## 🛠️ Installation MuJoCo for Linux

These steps set up MuJoCo 2.1.0 on Linux, including dependencies for headless rendering.

### Step 1: Install System Dependencies

```bash
sudo apt update

# Core graphics + GLFW
sudo apt install -y libxrandr2 libxinerama1 libxcursor1 libxi6 libgl1 libglfw3

# Headless rendering
sudo apt install -y libosmesa6 libosmesa6-dev xvfb libgl1-mesa-glx patchelf

# Build tools and XML/SSL support
sudo apt install -y build-essential python3-dev python3-pip \
    libglew-dev libssl-dev libffi-dev libxml2-dev libxslt1-dev zlib1g-dev
```

### Step 2: Install MuJoCo
Download and extract MuJoCo 2.1.0 to ~/.mujoco/mujoco210:

```bash
mkdir -p ~/.mujoco
cd ~/.mujoco
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar -xzf mujoco210-linux-x86_64.tar.gz
rm mujoco210-linux-x86_64.tar.gz
```

### Step 3: Set Environment Variables
Append the following to your `~/.bashrc` file:
```bash
# >>> MuJoCo environment setup >>>
export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu/nvidia
# <<< MuJoCo environment setup <<<
```
Apply the changes:
```bash
source ~/.bashrc
```

### Step 4: Reboot and Test Installation
After rebooting your system, run a simple MuJoCo simulation to verify setup:
```bash
cd ~/.mujoco/mujoco210/bin
./simulate ../model/humanoid.xml
```
If successful, you should see output like:
```bash
MuJoCo Pro version 2.10
ERROR: could not initialize GLFW
```
This error is expected on headless systems and confirms MuJoCo is correctly installed.
