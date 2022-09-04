---
layout: default
---

<center>Welcome to <b>C</b>ompute <b>G</b>eneral <b>L</b>ab (<b>CGL</b>)</center>

----

# CGL Research Areas

> A brief Record for the CGL's recent to do list (check [CGL SCHEDULE](./doc/arch/CGL_SCHEDULE.xlsx) for more detials and progress)

### [General Computing](./docs/arch/compute/computeIndex.html)
#### CPU ARCH [CG0](https://github.com/ComputeGeneral/CG0.git)
- CG0 RISC-V GV based Superscalar CPU Modeling.
- CG0 Compiler
  - Dragon book backend study

#### GPU ARCH [CG1](https://github.com/ComputeGeneral/CG1.git)
- [Compute API Notes](./docs/arch/apiIndex.html)
- [Compute Notes](./docs/arch/compute/computeIndex.md)
- CG1 Compute Specification
  - CG1 Scalar GPU ISA Specification( Memory Model Study)
  - SIMT Unified Shader Arch/MicroArch Specification (Work Distributer,fused Stream Processor)
- CG1 Modeling
  - High-level Analytical Performance Model(**PMDL**) and Energy Model(**EMDL**)
  - Highly abstracted behavior model (**BMDL**)
  - Low Level Algorithm Sharing between Functional CModel(**CMDL**) and Architecture/Cycle Model(**AMDL**)
  - HLS flow to generate Verilog Model (**VMDL**), and EDA flow for power and area estimation. (MatchLib integrate into CG1 )
- CG1 Compiler
  - Compiler backend code study

### [Compute Graphics](./docs/arch/graphics/graphicsIndex.html)
- [Graphics API Notes](./docs/arch/GraphicsApiIndex.html)
- [Graphics Notes](./docs/arch/graphics/graphicsIndex.html)
- CG1 Graphics Specification
  - Raster based module Specification
  - Ray Tracing based module Specification
- CG1 Modeling
  - Algorithm sharing between CMDL and AMDL.


### [Deep Learning](./docs/arch/deepLearning/deepLearningIndex.html)
- [DeepLearning Framework Notes](./docs/arch/DLFrameWorkIndex.html)
- [DeepLearning Notes](./docs/arch/deepLearning/deepLearningIndex.md)
- NVDLA $\rightarrow$ CG2


### Analysis TOOLs

- GPUVIS update for trace/statistics visualization
- API Trace and playback
- CMD Trace and playback system (Verification)
- Compute/Graphic Execution Trace
- Probe Statistics profiling (driver)
- R Language Study

---
