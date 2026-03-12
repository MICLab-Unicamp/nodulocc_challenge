# NoduLoCC2026 Challenge - Team MICLAB
Team members:
- Kristhian André Oliveira Aguilar (k298976@dac.unicamp.br)
- Diedre Santos do Carmo (diedre@unicamp.br)
- Letícia Rittner (lrittner@unicamp.br)

## Table of Contents
- [Method Summary](#method-summary)
    - [Classification Task – Fine-tuned MedGemma 1.5](#classification-task--fine-tuned-medgemma-15)
        - [Model](#model)
        - [Fine-tuning approach](#fine-tuning-approach)
        - [Addressing class imbalance](#addressing-class-imbalance)
        - [Training configuration](#training-configuration)
        - [Evaluation](#evaluation)
        - [Computational resources](#computational-resources)
    - [Localization Task – CheXagent-2 (Zero-shot)](#localization-task--chexagent-2-zero-shot)
        - [Model](#model-1)
        - [Pre-processing and inference pipeline](#pre-processing-and-inference-pipeline)
        - [Training configuration](#training-configuration-1)
        - [Computational resources](#computational-resources-1)
- [How to Run](#how-to-run)
    - [Classification Task — MedGemma 1.5](#classification-task-—-medgemma-15)
        - [Pre-requisites](#pre-requisites)
        - [Hugging Face API Token](#hugging-face-api-token)
        - [Environment Setup](#environment-setup)
        - [Inference](#inference)
        - [Training](#training)
    - [Localization Task — CheXagent-2 (Zero-shot)](#localization-task-—-chexagent-2-zero-shot)
        - [Pre-requisites](#pre-requisites-1)
        - [Environment Setup](#environment-setup-1)
        - [Inference](#inference-1)

## Method Summary

Vision-language models (VLMs) for chest X-ray analysis have rapidly become a strong research trend, especially for tasks such as report generation, abnormality classification, and localization. Motivated by this recent progress, our goal in this project was to evaluate whether modern medical VLMs can be effectively transferred to the NoduLoCC2026 challenge tasks, and how they behave under the practical constraints imposed by the challenge: severe class imbalance, limited localization annotations, and distribution shift between training and test data.

### Classification Task – Fine-tuned MedGemma 1.5
#### Model
MedGemma 1.5 is a 4B-parameter vision-language model developed by Google, built on the Gemma 3 architecture. Its vision component uses a SigLIP image encoder pre-trained on de-identified medical data including chest X-rays, while the LLM component was further trained on diverse medical corpora. All input images are normalized to 896×896 pixels. The model was selected over alternatives based on its best precision-recall tradeoff and F1 score at the 0.5 decision threshold during our model selection experiments.

#### Fine-tuning approach

We used QLoRA (Quantized Low-Rank Adaptation), which quantizes the base model weights to 4-bit NF4 precision and trains only a small set of low-rank adapter weights. This serves three purposes: (1) it reduces VRAM requirements significantly, allowing training on a 24GB consumer GPU; (2) it reduces the risk of overfitting on a relatively small dataset; and (3) it preserves the base VLM's general and medical knowledge by keeping most weights frozen.

#### Addressing class imbalance

The dataset is heavily imbalanced, with approximately 95% negative (healthy) and 5% positive (nodule) samples. To address this, we used two complementary strategies:

- **Weighted cross-entropy loss**: The loss function assigns a weight of 20× to positive samples and 1× to negatives, penalizing missed nodule detections much more heavily than false alarms.
- **Rotating balanced epoch sampler**: At each training epoch, all positive samples are included alongside a rotating, non-overlapping subset of negative samples of equal size. This ensures the model sees all negatives across epochs while maintaining a 1:1 class ratio within each epoch, avoiding both data waste and persistent imbalance.

#### Training configuration
- **Dataset split:** 80% training, 5% validation, 15% test (stratified by class).
- **Loss function:** Weighted cross-entropy (positive weight = 20, negative weight = 1)

- **Batch size:** Effective batch size of 16 (batch size of 4 with gradient accumulation over 4 steps)

- **Optimizer:** AdamW (fused)

- **Learning rate scheduler:** Linear warmup (3% of steps) followed by linear decay; peak LR = 2×10⁻⁴

#### Evaluation
For evaluation, we chose metrics that are robust to class imbalance and provide a comprehensive view of model performance. These include: Accuracy, balanced accuracy, sensitivity (recall), specificity, precision, F1, MCC, ROC-AUC, PR-AUC, Brier score.

The checkpoint chosen for final evaluation was the last one (step 4400). This checkpoint was selected based on its strong performance across multiple metrics, particularly its balanced accuracy and F1 score, which are crucial for imbalanced classification tasks.

The table below summarizes the performance of the fine-tuned MedGemma 1.5 model on the test set:

| Model | Precision | Recall | Specificity | F1    | ROC AUC | PR AUC |
| --------------- | --------- | ------ | ----------- | ----- | ------- | ------ |
| MedGemma 1.5    | 0.374     | 0.3621 | 0.9728      | 0.368 | 0.7864  | 0.2384 |

#### Computational resources
- **Parameters:** ~4B (base) + ~100M (QLoRA adapter)

- **Hardware:** NVIDIA GeForce RTX 4090 (24GB VRAM)

- **Training time:** ~35 hours

- **Inference time:** 2.5 seconds per image with batch size 1, using ~10GB VRAM on RTX 4090

### Localization Task – CheXagent-2 (Zero-shot)

#### Model
CheXagent-2 is a 3B-parameter vision-language model specialized for chest X-ray interpretation. It uses a fine-tuned SigLIP model as its vision encoder and a fine-tuned Phi-2 (2.7B) model as its language decoder, connected via a LLaVA-style MLP connector. The model was trained on CheXinstruct, a large-scale CXR dataset covering 35 tasks, and has demonstrated strong performance on localization and other CXR interpretation tasks. We use it zero-shot — no additional fine-tuning was performed.

#### Pre-processing and inference pipeline
Input images are clipped to [0.5, 99.5] pixel value percentiles and then normalized to [0, 255] in order to increase contrast in the image.

#### Training configuration
- *Loss function / Optimizer / LR scheduler:* N/A (zero-shot inference only)

- *Evaluation metrics:* **[FILL IN – e.g. IoU, localization accuracy, etc.]**

#### Computational resources
- *Parameters:* 3B
- *Hardware:* NVIDIA GeForce RTX 4090 (24GB VRAM)
- *Training time:* N/A
- *Inference time:* 1 second per image with batch size 1, using ~10GB VRAM on RTX 4090

## How to Run
### Classification Task — MedGemma 1.5 
#### Pre-requisites
- Python 3.12
- Environment manager (e.g., Conda, Mamba)
- Hugging Face account, MedGemma 1.5 license acknowledgment, and API token with read permissions
- GPU with 10GB+ for training and inference. The GPU used for training was an NVIDIA GeForce RTX 4090 with 24GB of VRAM.

#### Hugging Face API Token
To download the MedGemma 1.5 model, you need to have a Hugging Face account and accept the license agreement for MedGemma 1.5. 

If you already have a Hugging Face account and a Hugging Face API token with read permissions, just acknowledge the license in https://huggingface.co/google/medgemma-1.5-4b-it and add the token to your `.env` file as `HF_TOKEN=your_token_here`. See the `.env.example` file for reference. 

If you don't have a Hugging Face account or don't have an API token, you can create an account and generate a token for free. To do this, [watch this video tutorial](https://youtu.be/IB4oYDhhMEk) or follow these steps:
1. Create a Hugging Face account if you don't have one: https://huggingface.co/join
2. **After confirming your email** and logging in, go to the MedGemma 1.5 model page: https://huggingface.co/google/medgemma-1.5-4b-it
3. Click on the "Acknowledge License" button to accept the license agreement. **This will not work if you are not signed in or if you haven't confirmed your email address.**
4. Once you have accepted the license, go to your Hugging Face account settings: https://huggingface.co/settings/tokens
5. Click on "Create new token", give it a name (e.g., "MedGemma Access"), select the "Read" role, and click "Generate".
6. Copy the generated token and add it to your `.env` file as `HF_TOKEN=your_token_here`. See the `.env.example` file for reference.

#### Environment Setup
The environment can be set up using the following commands:
```bash
conda create -n nodulocc python=3.12 -y
conda activate nodulocc
pip install -r classification_task/requirements.txt
```

#### Inference
```bash
python classification_task/inference.py --input-dir ./data/test_images --output-dir ./outputs
```

#### Training
Requires the nodulocc dataset to be downloaded and placed in the `data/` directory. The dataset should be organized as follows:
```
data/nodulocc/
├── lidc_png_16_bit/
├── nih_filtered_images
├── classification_labels.csv
└── localization_labels.csv
```

Then, run the training script:
```bash
python classification_task/train.py
```

By default the training script will fine-tune the MedGemma 1.5 model from scratch. In case you want to continue training from a checkpoint, modify the `resume_from_checkpoint` argument in `trainer.train()` to point to the desired checkpoint directory. 

For example:
```python
trainer.train(resume_from_checkpoint="./checkpoints/medgemma-1.5-nodulocc-cls-ckpt")
```

[You can get the model checkpoint from this Google Drive link](https://drive.google.com/file/d/1I6tu9mFlv2b3Y4LVV_FIqSocZBvRnMkH/view?usp=sharing). This checkpoint was trained for 4400 steps.

### Localization Task — CheXagent-2 (Zero-shot)
#### Pre-requisites
- Python 3.10
- Environment manager (e.g., Conda, Mamba)
- GPU with 8GB+ for inference. The GPU used for training was an NVIDIA GeForce RTX 4090 with 24GB of VRAM.

#### Environment Setup
The environment can be set up using the following commands:
```bash
conda create -n chexagent2 python=3.10 -y
conda activate chexagent2
pip install -r localization_task/requirements.txt
```

#### Inference
```bash
python localization_task/inference.py --input-dir ./data/test_images --output-dir ./outputs