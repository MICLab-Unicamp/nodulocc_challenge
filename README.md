
## Classification Task — MedGemma 1.5
### Pre-requisites
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

### Inference
```bash
python classification_task/inference.py --input-dir ./data/test_images --output-dir ./outputs
```

### Training
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

[You can get the model checkpoint from this Google Drive link](https://drive.google.com/file/d/1I6tu9mFlv2b3Y4LVV_FIqSocZBvRnMkH/view?usp=sharing). This checkpoint was trained for 12 epochs using the same training script.

## Localization Task — MedGemma 1.5
### Pre-requisites
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

### Inference
```bash
python localization_task/inference.py --input-dir ./data/test_images --output-dir ./outputs