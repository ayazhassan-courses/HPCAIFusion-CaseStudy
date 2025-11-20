# HPCAIFusion-CaseStudy
This study investigates the fusion of High-Performance Computing (HPC) and Scientific Machine Learning (SciML) using the 2D transient heat equation as a benchmark. A research-oriented comparative analysis will be carried out to study accuracy, runtime, scalability, and the potential advantages of HPC–AI integration.

# 📘 **HPC + AI Fusion: Scientific Machine Learning with GPUs**

### *2D Heat Equation — CUDA Stencil Solver + Physics-Informed Neural Network (PINN)*

**Course:** COE-509 — Special Topics in HPC
**Instructor:** Dr. Ayaz ul Hassan Khan

---

## 🚀 **Study Overview**

This study explores the fusion of **High-Performance Computing (HPC)** and **Scientific Machine Learning (SciML)** using the **2D transient heat equation** as a case study.

You will:

1. Implement and optimize a **GPU stencil solver** using CUDA.
2. Build a **Physics-Informed Neural Network (PINN)** using PyTorch/JAX.
3. Compare both methods in terms of:

   * accuracy
   * runtime
   * scaling
   * GPU efficiency (roofline analysis)

This is a **research-style study**. You are expected to think critically, justify design choices, and provide clear experiments.

---

## 📁 **Repository Structure**

```text
hpc-ai-heat2d/
├── cuda_solver/
│   ├── include/            # C++ headers (API for GPU solver)
│   ├── src/                # CUDA kernels + main program
│   ├── CMakeLists.txt      # CUDA build configuration
│
├── pinn/
│   ├── models.py           # PINN architecture (MLP)
│   ├── datasets.py         # Sampling PDE/IC/BC points
│   ├── train_pinn.py       # Main training script
│   ├── evaluate_pinn.py    # Evaluate PINN on grid
│   └── config.yaml         # Hyperparameters, paths, settings
│
├── analysis/
│   ├── compare_solutions.py    # Compare CUDA vs PINN outputs
│   ├── plot_roofline_stub.py   # Starter roofline plotting script
│   └── utils_io.py             # Simple read/write helpers
│
├── data/
│   ├── cuda_reference/     # Solver outputs (snapshots, final fields)
│   └── pinn_outputs/       # PINN model + predictions
│
├── docs/
│   └── study_description.pdf
│
└── README.md               # (this file)
```

---

## 📌 **Tasks Summary**

### **Phase 0 — Numerical Derivations**

* Derive 5-point explicit stencil
* Derive CFL condition
* Compute arithmetic intensity

Deliverable: short PDF report.

---

### **Phase 1 — GPU Stencil Solver (CUDA)**

You will fill in:

* `heat2d_kernel.cu` → **write the CUDA stencil kernel**
* `run_heat2d_gpu()` → **launch + manage simulation**
* Add shared-memory optimization (optional / advanced)

You must:

* Compare naive vs optimized kernels
* Measure runtime vs grid size
* Produce a **roofline plot** placing your achieved performance

Outputs go in: `data/cuda_reference/`.

---

### **Phase 2 — PINN Implementation**

You will fill in:

* `datasets.py` → sampling interior, IC, BC points
* `heat_pde_residual()` → compute PDE residual using autograd
* `train_pinn.py` → training loop (logging, checkpoints)

You must:

* Train a PINN on GPU
* Plot training curves
* Compare predictions to CUDA solver

Outputs go in: `data/pinn_outputs/`.

---

### **Phase 3 — Comparative Analysis**

Use `analysis/compare_solutions.py` to produce:

* Heatmaps of both solutions
* Error maps
* L2/relative error metrics
* **Error vs Runtime** plot (required)

Use `analysis/plot_roofline_stub.py` to produce:

* Your roofline figure

Deliverable: **3–5 page research-style report**.

---

### Optional (Phase 4)

You may extend the study with:

* Multi-GPU stencil solver
* Multi-GPU PINN training
* Varying thermal diffusivity
* Complex BCs
* UQ for PINNs
* Neural operator instead of PINN

Extra credit awarded for innovation.

---

## 🛠️ **Environment Setup**

### **Prerequisites**

* CUDA 11+
* Python 3.9+
* PyTorch (with CUDA support) or JAX
* Nsight Systems / Nsight Compute
* CMake ≥ 3.18

---

### **Build CUDA Solver**

```bash
cd cuda_solver
mkdir build && cd build
cmake ..
make -j
./heat2d
```

Outputs saved to:

```
../data/cuda_reference/
```

---

### **Train PINN**

Edit hyperparameters in:

```
pinn/config.yaml
```

Then run:

```bash
cd pinn
python train_pinn.py
```

Outputs saved to:

```
../data/pinn_outputs/
```

---

### **Compare Results**

```bash
cd analysis
python compare_solutions.py
```

---

## 📊 **Required Plots**

You must include:

* Roofline diagram with measured GPU performance
* CUDA solution heatmap & PINN heatmap
* Error heatmap
* Training curves for PINN (loss vs epoch)
* **Error vs Runtime**: CUDA vs PINN (key figure)
* Scaling results

---

## 🧪 **What You Should NOT Commit**

* Large binary files (>20MB)
* Virtual environment folders
* Auto-generated PyTorch checkpoints (except final model)
* Build folders (`cuda_solver/build`)

---

## 📚 **Academic Integrity**

* Discussion of ideas is allowed.
* All code, analysis, and writing must be your own.
* Auto-generated AI code without understanding is not permitted.

---

## 🙋 Need Help?

Ask clarifying questions on the MS-Teams/Email or during office hours.
However, debugging **your own** code is part of the learning process.

---
