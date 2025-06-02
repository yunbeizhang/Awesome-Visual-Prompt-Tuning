# Awesome Visual Prompt Tuning
[![Awesome Badges](https://img.shields.io/badge/badges-awesome-green.svg)](https://github.com/Naereen/badges)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)]()
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

A curated list of awesome papers, resources, and tools for Visual Prompt Tuning (VPT).

---
Welcome to the **Awesome Visual Prompt Tuning** repository! This collection aims to be a comprehensive resource for researchers and practitioners interested in the rapidly evolving field of Visual Prompt Tuning (VPT). VPT offers a parameter-efficient way to adapt large pre-trained vision models to downstream tasks by introducing and optimizing small sets of prompt parameters, rather than fine-tuning the entire model.

We have organized the surveyed papers into three primary categories based on the nature of the prompts:

1.  [**Learnable Visual Prompts**](#learnable-visual-prompts)
2.  **Generative Visual Prompts:** These prompts are dynamically generated, often by another model or process.
3.  **Non-Learnable Visual Prompts:** These prompts are typically handcrafted or based on predefined transformations.

Within each of these main categories, we further distinguish between two levels of prompt application:

* **Pixel-Level Prompts:** Modifications are made directly to the input image pixels.
* **Token-Level Prompts:** Prompts are introduced at the level of image patches or feature tokens, often interacting with the model's internal representations (e.g., in Vision Transformers).

Beyond this core categorization, we also highlight the application and exploration of VPT in several significant and emerging areas:

1.  **VPT at Test Time (Test-Time Adaptation):** Adapting models to new data encountered during inference.
2.  **Grey/Black-Box VPT:** Applying VPT techniques when access to model weights or gradients is limited or unavailable.
3.  **VPT in Multi-Modality:** Exploring how visual prompts interact and integrate with other modalities like text or audio.
4.  **VPT for Trustworthy AI:** Investigating the role of VPT in enhancing aspects such as:
    * **Privacy:** Protecting sensitive information in visual data.
    * **Robustness:** Improving model performance against adversarial attacks or distribution shifts.
    * **Fairness:** Mitigating biases in model predictions.
    * **Calibration:** Ensuring model confidence aligns with prediction accuracy.
    * **Domain Generalization:** Improving model performance on unseen domains.

We encourage contributions and discussions to keep this repository up-to-date and comprehensive.

---

## Learnable Visual Prompts
<a name="learnable-visual-prompts"></a>

This subsection focuses on visual prompts that consist of parameters optimized during the tuning process.

| Title | Venue | Year | Keywords/Summary |
| :---- | :---- | :--- | :--------------- |
|  [Adversarial Reprogramming of Neural Networks](https://arxiv.org/abs/1806.11146)     |   ICLR    |  2019    |    First work            |
| [Transfer Learning without Knowing: Reprogramming Black-box Machine Learning Models with Scarce Data and Limited Resources](https://arxiv.org/abs/2007.08714) | ICML | 2020 | [Code](https://github.com/yunyuntsai/Black-box-Adversarial-Reprogramming), Black-box|
| [An Improved (Adversarial) Reprogramming Technique for Neural Networks](https://link.springer.com/chapter/10.1007/978-3-030-86362-3_1) | ICANN | 2021 | |
|[Adversarial Reprogramming of Pretrained Neural Networks for Fraud Detection](https://dl.acm.org/doi/abs/10.1145/3459637.3482053) | CIKM| 2021 | |
|[Fairness Reprogramming](https://arxiv.org/pdf/2209.10222) | NeurIPS | 2022 | [Code](https://github.com/UCSB-NLP-Chang/Fairness-Reprogramming), Fairness|
| [Bayesian-guided Label Mapping for Visual Reprogramming](https://arxiv.org/abs/2410.24018) |  NeurIPS | 2024 | [Code](https://github.com/tmlr-group/bayesianlm), Label mapping|
| [Sample-specific Masks for Visual Reprogramming-based Prompting](https://arxiv.org/abs/2406.03150) | ICML | 2024 | [Code](https://github.com/tmlr-group/SMM)|
| [Attribute-based Visual Reprogramming for Vision-Language Models](https://arxiv.org/abs/2501.13982) | ICLR | 2025 | [Code](https://github.com/tmlr-group/attrvr), Multi-modality|
| [AutoVP: An Automated Visual Prompting Framework and Benchmark](https://arxiv.org/abs/2310.08381) | ICLR | 2024 | [Code](https://github.com/IBM/AutoVP)|
| [Understanding and Improving Visual Prompting: A Label-Mapping Perspective](https://arxiv.org/abs/2211.11635) | CVPR | 2023 | [Code](https://github.com/optml-group/ilm-vp)|