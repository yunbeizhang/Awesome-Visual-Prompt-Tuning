<h1 align="center">Awesome Visual Prompt Tuning</h1>

<div align="center">

[![Awesome Badges](https://img.shields.io/badge/badges-awesome-green.svg)](https://github.com/Naereen/badges)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)]()
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

*A curated list of awesome papers, resources, and tools for Visual Prompt Tuning (VPT).*
</div>



---

Welcome to the **Awesome Visual Prompt Tuning** repository! This collection aims to be a comprehensive resource for researchers and practitioners interested in the rapidly evolving field of Visual Prompt Tuning (VPT). VPT offers a parameter-efficient way to adapt large pre-trained vision models to downstream tasks by introducing and optimizing small sets of prompt parameters, rather than fine-tuning the entire model.

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Core Methodologies](#core-methodologies)
  - [1. Learnable Visual Prompts](#1-learnable-visual-prompts)
    - [Pixel-Level](#pixel-level)
    - [Token-Level](#token-level)
  - [2. Generative Visual Prompts](#2-generative-visual-prompts)
    - [Pixel-Level](#pixel-level-1)
    - [Token-Level](#token-level-1)
  - [3. Non-Learnable Visual Prompts](#3-non-learnable-visual-prompts)
- [Applications and Advanced Topics](#applications-and-advanced-topics)
  - [VPT at Test-Time (Test-Time Adaptation)](#vpt-at-test-time-test-time-adaptation)
  - [Grey/Black-Box VPT](#greyblack-box-vpt)
  - [VPT in Multimodality](#vpt-in-multimodality)
  - [VPT for Trustworthy AI](#vpt-for-trustworthy-ai)
    - [Robustness](#robustness)
    - [Fairness](#fairness)
    - [Privacy](#privacy)
    - [Calibration](#calibration)
    - [Domain Generalization](#domain-generalization)
- [Related Surveys and Benchmarks](#related-surveys-and-benchmarks)
---

## Core Methodologies
<a name="core-methodologies"></a>
We have organized the primary surveyed papers into three categories based on the nature of the prompts.

### 1. Learnable Visual Prompts
<a name="learnable-visual-prompts"></a>
This section covers prompts that consist of learnable parameters optimized during the tuning process.

#### Pixel-Level
*Modifications are made directly to the input image pixels.*
| Title | Venue | Year | Keywords |
| :---- | :---- | :--- | :--------------- |
| [Adversarial Reprogramming of Neural Networks](https://arxiv.org/abs/1806.11146) | ICLR | 2019 | First work |
| [Transfer Learning without Knowing: Reprogramming Black-box Machine Learning Models with Scarce Data and Limited Resources](https://arxiv.org/abs/2007.08714) | ICML | 2020 | [Code](https://github.com/yunyuntsai/Black-box-Adversarial-Reprogramming), Black-box |
| [An Improved (Adversarial) Reprogramming Technique for Neural Networks](https://link.springer.com/chapter/10.1007/978-3-030-86362-3_1) | ICANN | 2021 | - |
| [Adversarial Reprogramming of Pretrained Neural Networks for Fraud Detection](https://dl.acm.org/doi/abs/10.1145/3459637.3482053) | CIKM | 2021 | - |
| [Fairness Reprogramming](https://arxiv.org/pdf/2209.10222) | NeurIPS | 2022 | [Code](https://github.com/UCSB-NLP-Chang/Fairness-Reprogramming), Fairness |
| [Understanding and Improving Visual Prompting: A Label-Mapping Perspective](https://arxiv.org/abs/2211.11635) | CVPR | 2023 | [Code](https://github.com/optml-group/ilm-vp), Label mapping |
| [AutoVP: An Automated Visual Prompting Framework and Benchmark](https://arxiv.org/abs/2310.08381) | ICLR | 2024 | [Code](https://github.com/IBM/AutoVP) |
| [Bayesian-guided Label Mapping for Visual Reprogramming](https://arxiv.org/abs/2410.24018) | NeurIPS | 2024 | [Code](https://github.com/tmlr-group/bayesianlm), Label mapping |
| [Sample-specific Masks for Visual Reprogramming-based Prompting](https://arxiv.org/abs/2406.03150) | ICML | 2024 | [Code](https://github.com/tmlr-group/SMM) |
| [Attribute-based Visual Reprogramming for Vision-Language Models](https://arxiv.org/abs/2501.13982) | ICLR | 2025 | [Code](https://github.com/tmlr-group/attrvr), Multi-modal |
| [Visual Prompting for Adversarial Robustness](https://arxiv.org/abs/2210.06284) | ICASSP | 2023 | [Code](https://github.com/Phoveran/vp-for-adversarial-robustness), Adversarial Robustness|
| [Understanding Zero-Shot Adversarial Robustness for Large-Scale Models](https://arxiv.org/abs/2212.07016) | ICLR | 2023 | [Code](https://github.com/cvlab-columbia/ZSRobust4FoundationModel), Adversarial Robustness|
| [One Prompt Word is Enough to Boost Adversarial Robustness for Pre-trained Vision-Language Models](https://arxiv.org/pdf/2403.01849) | CVPR | 2024 | [Code](https://github.com/TreeLLi/APT), Adversarial Robustness, Multi-modal|
| [Exploring the Benefits of Visual Prompting in Differential Privacy](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Exploring_the_Benefits_of_Visual_Prompting_in_Differential_Privacy_ICCV_2023_paper.pdf) | ICCV | 2023 | [Code](https://github.com/EzzzLi/Prom-PATE), Privacy|
| [Neural Clamping: Joint Input Perturbation and Temperature Scaling for Neural Network Calibration](https://arxiv.org/pdf/2209.11604) | TMLR | 2024 | [Code](https://github.com/yungchentang/NCToolkit), Uncertainty|
| [Unleashing the Power of Visual Prompting At the Pixel Level](https://arxiv.org/pdf/2212.10556) | TMLR | 2024 | [Code](https://github.com/UCSC-VLAA/EVP)|

#### Token-Level
*Prompts are introduced at the level of feature tokens within the model.*
| Title | Venue | Year | Keywords |
| :---- | :---- | :--- | :--------------- |
| [Visual Prompt Tuning](https://link.springer.com/chapter/10.1007/978-3-031-19827-4_41) | ECCV | 2022 | [Github](https://github.com/KMnP/vpt) |
| [LPT: Long-tailed Prompt Tuning for Image Classification](https://arxiv.org/abs/2210.01033) | ICLR | 2023 | [Github](https://github.com/DongSky/LPT) |
| [Learning Expressive Prompting With Residuals for Vision Transformers](https://arxiv.org/abs/2303.15591) | CVPR | 2023 | - |
| [Improving Visual Prompt Tuning for Self-supervised Vision Transformers](https://arxiv.org/abs/2306.05067) | ICML | 2023 | [Github](https://github.com/ryongithub/GatedPromptTuning) |
| [E^2VPT: An Effective and Efficient Approach for Visual Prompt Tuning](https://arxiv.org/pdf/2307.13770) | ICCV | 2023 | [Github](https://github.com/ChengHan111/E2VPT) |
| [SA2VP: Spatially Aligned-and-Adapted Visual Prompt](https://arxiv.org/abs/2312.10376) | AAAI | 2024 | [Github](https://github.com/tommy-xq/SA2VP) |
| [Revisiting the Power of Prompt for Visual Tuning](https://arxiv.org/abs/2402.02382) | ICML | 2024 | [Github](https://github.com/WangYZ1608/Self-Prompt-Tuning) |
| [Adaptive Prompt: Unlocking the Power of Visual Prompt Tuning](https://arxiv.org/abs/2411.01327) | arXiv | 2025 | [Github](https://github.com/runtsang/VFPT) |
| [Semantic-Guided Visual Prompt Tuning for Vision Transformers](https://arxiv.org/abs/2505.23694) | CVPR | 2025 | [Github](https://github.com/runtsang/VFPT) |
| [Understanding Zero-Shot Adversarial Robustness for Large-Scale Models](https://arxiv.org/abs/2212.07016) | ICLR | 2023 | [Code](https://github.com/cvlab-columbia/ZSRobust4FoundationModel), (Pixel and Token)Adversarial Robustness|

### 2. Generative Visual Prompts
<a name="generative-visual-prompts"></a>
This section includes prompts that are dynamically generated, often by another model or process.

#### Pixel-Level
| Title | Venue | Year | Keywords |
| :---- | :---- | :--- | :--------------- |
|       |       |      |                  |

#### Token-Level
| Title | Venue | Year | Keywords |
| :---- | :---- | :--- | :--------------- |
|       |       |      |                  |


### 3. Non-Learnable Visual Prompts
<a name="non-learnable-visual-prompts"></a>
This section focuses on prompts that are handcrafted or based on predefined transformations, without learnable parameters.

| Title | Venue | Year | Keywords |
| :---- | :---- | :--- | :--------------- |
|       |       |      |                  |

---

## Applications and Advanced Topics
<a name="applications-and-advanced-topics"></a>
Beyond the core methodologies, this section highlights the application of VPT in significant and emerging areas.

### VPT at Test-Time (Test-Time Adaptation)
<a name="vpt-at-test-time-test-time-adaptation"></a>
*Papers focusing on adapting models using prompts on new data encountered during inference. This setting is particularly challenging as it is often fully **unsupervised**, requiring adaptation without access to ground-truth labels.*
| Title | Venue | Year | Keywords |
| :---- | :---- | :--- | :--------------- |
|       |       |      |                  |

### Grey/Black-Box VPT
<a name="greyblack-box-vpt"></a>
*Applying VPT techniques when access to model weights or gradients is limited or unavailable. A key feature here is the reliance on **gradient-free** methods (e.g., Zeroth-Order Optimization - ZOO) to optimize prompts when backpropagation is not possible.*
| Title | Venue | Year | Keywords |
| :---- | :---- | :--- | :--------------- |
|       |       |      |                  |

### VPT in Multimodality
<a name="vpt-in-multimodality"></a>
*Exploring how visual prompts interact and integrate with other modalities like text or audio.*
| Title | Venue | Year | Keywords |
| :---- | :---- | :--- | :--------------- |
|       |       |      |                  |

### VPT for Trustworthy AI
<a name="vpt-for-trustworthy-ai"></a>
Investigating the role of VPT in enhancing different aspects of model trustworthiness.

#### Robustness
<a name="robustness"></a>
*Improving model performance against adversarial attacks or distribution shifts.*
| Title | Venue | Year | Keywords |
| :---- | :---- | :--- | :--------------- |
|       |       |      |                  |

#### Fairness
<a name="fairness"></a>
*Mitigating biases in model predictions across different demographic groups.*
| Title | Venue | Year | Keywords |
| :---- | :---- | :--- | :--------------- |
|       |       |      |                  |

#### Privacy
<a name="privacy"></a>
*Protecting sensitive information in visual data during model training or inference.*
| Title | Venue | Year | Keywords |
| :---- | :---- | :--- | :--------------- |
|       |       |      |                  |

#### Calibration
<a name="calibration"></a>
*Ensuring model prediction confidence aligns with prediction accuracy.*
| Title | Venue | Year | Keywords |
| :---- | :---- | :--- | :--------------- |
|       |       |      |                  |

#### Domain Generalization
<a name="domain-generalization"></a>
*Improving model performance on unseen domains and distributions.*
| Title | Venue | Year | Keywords |
| :---- | :---- | :--- | :--------------- |
|       |       |      |                  |

---

## Related Surveys and Benchmarks
<a name="related-surveys-and-benchmarks"></a>
This section lists other relevant survey papers and benchmarks in the broader area of prompt-based learning and parameter-efficient fine-tuning.

| Title | Venue | Year | Keywords |
| :---- | :---- | :--- | :--------------- |
| [A Systematic Survey of Prompt Engineering on Vision-Language Foundation Models](https://arxiv.org/abs/2307.12980)      |       |      |    [Code](https://github.com/JindongGu/Awesome-Prompting-on-Vision-Language-Model)              |
| [Parameter-Efficient Fine-Tuning for Pre-Trained Vision Models: A Survey](https://arxiv.org/abs/2402.02242) | | | [Code](https://github.com/synbol/awesome-parameter-efficient-transfer-learning) |
| [Prompt learning in computer vision: a survey](https://link.springer.com/content/pdf/10.1631/FITEE.2300389.pdf) | | | |
