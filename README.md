<h1 align="center">Awesome Prompt-Based Adaptation for Vision Models</h1>

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2510.13219-b31b1b.svg?style=flat-square)](https://arxiv.org/pdf/2510.13219)
[![Awesome Badges](https://img.shields.io/badge/badges-awesome-green.svg?style=flat-square)](https://github.com/Naereen/badges)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg?style=flat-square)]()
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)


*A curated list of papers, resources and tools on **Prompt-Based Adaptation (PA)** for large-scale vision models.*
</div>

---

## Introduction and Motivation

Large vision models, such as Vision Transformers and convolutional backbones, are typically pretrained on massive datasets and then finetuned for downstream tasks. Finetuning all parameters is expensive and may erode pretrained knowledge. **Prompt-Based Adaptation (PA)** introduces small prompt parameters while freezing the backbone — steering pretrained models efficiently to new tasks.  
The survey *“Prompt-based Adaptation in Large-scale Vision Models: A Survey”* defines PA as a unified framework covering both **Visual Prompting (VP)** and **Visual Prompt Tuning (VPT)**.  
- **VP** modifies the input image via pixel-space prompts.  
- **VPT** injects learnable tokens inside the network.  

Both achieve adaptation with minimal parameter updates and strong generalization.

---

## Table of Contents

- [Unified Taxonomy](#unified-taxonomy)
  - [Visual Prompting (VP)](#visual-prompting-vp)
    - [VP-Fixed](#vp-fixed)
    - [VP-Learnable](#vp-learnable)
    - [VP-Generated](#vp-generated)
  - [Visual Prompt Tuning (VPT)](#visual-prompt-tuning-vpt)
    - [VPT-Learnable](#vpt-learnable)
    - [VPT-Generated](#vpt-generated)
  - [Efficiency Considerations](#efficiency-considerations)
- [Applications Across Vision Tasks](#applications-across-vision-tasks)
  - [Segmentation](#segmentation)
  - [Restoration & Enhancement](#restoration--enhancement)
  - [Compression](#compression)
  - [Multi-Modal Tasks](#multi-modal-tasks)
- [Domain-Specific Applications](#domain-specific-applications)
  - [Medical & Biomedical Imaging](#medical--biomedical-imaging)
  - [Remote Sensing & Geospatial Analysis](#remote-sensing--geospatial-analysis)
  - [Robotics & Embodied AI](#robotics--embodied-ai)
  - [Industrial Inspection & Manufacturing](#industrial-inspection--manufacturing)
  - [Autonomous Driving & ADAS](#autonomous-driving--adas)
  - [3D Point Clouds & LiDAR](#3d-point-clouds--lidar)
- [PA under Practical Constraints](#test-time-and-resource-constrained-adaptation)
- [Trustworthy AI](#trustworthy-ai)
- [Related Surveys and Benchmarks](#related-surveys-and-benchmarks)
- [Contributing](#contributing)

---

## Unified Taxonomy

PA methods are categorized by *where* prompts are injected (input vs. token space) and *how* they’re obtained (fixed, learnable, generated).

### Visual Prompting (VP)
Prompts are applied directly to pixels before tokenization:
\[
\tilde{x} = u(x;\theta)
\]
- **VP-Fixed**: no learnable parameters — static boxes, points, or masks (e.g., SAM).  
- **VP-Learnable**: optimize pixel-space overlays, frequency cues, or masks (e.g., Fourier VP, OT-VP).  
- **VP-Generated**: a generator produces adaptive image-level prompts (e.g., BlackVIP).

| Title | Venue | Year | Type | Notes |
|:------|:------|:------|:------|:------|
| [Fourier Visual Prompting](https://arxiv.org/abs/2212.10556) | TMLR | 2024 | Learnable | Frequency-domain cues |
| [BlackVIP](https://arxiv.org/abs/2303.14773) | CVPR | 2023 | Generated | Zeroth-order black-box |
| [Custom SAM](https://arxiv.org/abs/2306.05385) | 2023 | Learnable | Medical segmentation |
| [Insight Any Instance](https://arxiv.org/abs/2402.03771) | 2025 | Learnable | Remote sensing |
| [Visual Prompting via Inpainting](https://arxiv.org/abs/2203.17274) | NeurIPS | 2022 | Generated | Early adaptive VP |

### Visual Prompt Tuning (VPT)

VPT inserts learnable tokens into frozen model layers:
\[
Z^{(\ell)} = [x_{cls}; P^{(\ell)}; x_1; …; x_N]
\]
- **VPT-Learnable**: prompt tokens are trained via gradient descent.  
- **VPT-Generated**: small networks produce adaptive prompt tokens.

| Title | Venue | Year | Type | Notes |
|:------|:------|:------|:------|:------|
| [VPT](https://arxiv.org/abs/2203.17274) | ECCV | 2022 | Learnable | Foundational method |
| [LPT](https://arxiv.org/abs/2210.01033) | ICLR | 2023 | Learnable | Long-tailed classes |
| [SA2VP](https://arxiv.org/abs/2312.10376) | AAAI | 2024 | Learnable | Spatially aligned 2D map |
| [E2VPT](https://arxiv.org/abs/2307.13770) | ICCV | 2023 | Learnable | Key–value prompts |
| [DVPT](https://arxiv.org/abs/2505.04119) | NN | 2025 | Generated | Cross-attention generator |

---

## Applications Across Vision Tasks

### Segmentation
Prompts help continual, multimodal, and few-shot segmentation (e.g., SAM-adapters, SA2VP).

### Restoration & Enhancement
PromptIR and PromptRestorer inject degradation-aware prompts for denoising, dehazing, deraining, etc.

### Compression
Prompt tokens control rate–distortion trade-offs in Transformer codecs and guide semantic compression in video.

### Multi-Modal Tasks
Visual prompts condition multimodal models (CLIP, MLLMs) to refine image-language alignment and visual reasoning.

---

## Domain-Specific Applications

### Medical & Biomedical Imaging
Prompted SAM variants (CusSAM, Ma-SAM) adapt foundation models for 2D/3D medical segmentation and reporting.  
VPT bridges visual–textual reasoning for clinical report generation.

### Remote Sensing & Geospatial
RSPrompter, ZoRI, and PHTrack apply prompts for segmentation, change detection, and hyperspectral analysis.

### Robotics & Embodied AI
Prompts adapt 2D backbones for 3D or motion reasoning (e.g., PointCLIP, ShapeLLM, GAPrompt).

### Industrial Inspection
Prompts steer SAM/CLIP for zero-shot defect segmentation and anomaly detection (e.g., ClipSAM, SAID).

### Autonomous Driving & ADAS
Severity-aware and differentiable prompts improve perception in adverse conditions, with minimal retraining.

### 3D Point Clouds & LiDAR
Token-level prompts enhance geometric reasoning and fusion in LiDAR–camera systems (e.g., PointLoRA, PromptDet).

---

## Test-Time and Resource-Constrained Adaptation

Prompts enable on-the-fly adaptation to unseen domains:
- **TTA**: test-time prompt tuning (e.g., DynaPrompt, C-TPT).  
- **Black-Box**: zeroth-order learning (e.g., BlackVIP).  
- **Federated / Source-Free**: decentralized personalized prompts (e.g., FedPrompt, DDFP).

---

## Trustworthy AI

PA contributes to **robustness**, **fairness**, and **privacy**:
- Robust prompts improve adversarial resistance.  
- Fairness prompts mitigate demographic bias.  
- Privacy prompts protect sensitive visual data.  
- Calibration aligns confidence with accuracy.

---

## Related Surveys and Benchmarks

| Title | Venue | Year | Notes |
|:------|:------|:------|:------|
| [Prompt Learning in Computer Vision: A Survey](https://link.springer.com/content/pdf/10.1631/FITEE.2300389.pdf) | FITEE | 2024 | General overview |
| [Parameter-Efficient Fine-Tuning for Pre-Trained Vision Models](https://arxiv.org/abs/2402.02242) | arXiv | 2024 | PEFT methods |
| [Prompt Engineering on Vision-Language Models](https://arxiv.org/abs/2307.12980) | arXiv | 2023 | VL prompts |
| [Visual Prompting in MLLMs](https://arxiv.org/abs/2409.15310) | arXiv | 2024 | MLLM prompts |

---

## Contributing

We welcome new papers, implementations, and corrections!  
Please categorize contributions under:
- VP-Fixed / Learnable / Generated  
- VPT-Learnable / Generated  
- And note the application domain (e.g., Medical, 3D, Remote Sensing).

---

## Citation
If you find this survey useful in your research, please consider citing our paper:

```bibtex
@article{xiao2025prompt,
  title={Prompt-based Adaptation in Large-scale Vision Models: A Survey},
  author={Xiao, Xi and Zhang, Yunbei and Zhao, Lin and Liu, Yiyang and Liao, Xiaoying and Mai, Zheda and Li, Xingjian and Wang, Xiao and Xu, Hao and Hamm, Jihun and others},
  journal={arXiv preprint arXiv:2510.13219},
  year={2025}
}
```
