# Awesome Visual Prompt Tuning
[![Awesome Badges](https://img.shields.io/badge/badges-awesome-green.svg)](https://github.com/Naereen/badges) 
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)]()
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)


A curated list of awesome papers, resources, and tools for Visual Prompt Tuning (VPT).

---
Welcome to the **Awesome Visual Prompt Tuning** repository! This collection aims to be a comprehensive resource for researchers and practitioners interested in the rapidly evolving field of Visual Prompt Tuning (VPT). VPT offers a parameter-efficient way to adapt large pre-trained vision models to downstream tasks by introducing and optimizing small sets of prompt parameters, rather than fine-tuning the entire model.

We have organized the surveyed papers into three primary categories based on the nature of the prompts:

1.  **Non-Learnable Visual Prompts:** These prompts are typically handcrafted or based on predefined transformations.
2.  **Learnable Visual Prompts:** These prompts consist of parameters that are optimized during the tuning process.
3.  **Generative Visual Prompts:** These prompts are dynamically generated, often by another model or process.

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

