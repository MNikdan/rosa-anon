# RoSA

This repository contains the code for the paper "RoSA: Accurate Parameter-Efficient Fine-Tuning via Robust Adaptation."

# Installation
Run the following commands to install the required packages:
```
conda create --name rosa python=3.10 -y
conda activate rosa
cd llmfoundry && pip install -e . && cd ..
cd lm-evaluation-harness && pip install -e . && cd ..
cd spops && pip install ninja && pip install -e . && cd ..
cd peft-rosa && pip install -e . && cd ..
pip install -U datasets bitsandbytes fire
```

# Training
```
cd llmfoundry/scripts/train/

// choose one of the scripts below, or edit one to get your desired config
bash scripts/restart_7b_rosa_gsm_bf16.sh
bash scripts/restart_7b_rosa_gsm_4bit.sh
bash scripts/restart_13b_rosa_gsm_bf16.sh
```

# Evaluation
The training scripts will automatically run the evaluations as well (look at the final lines of the scripts), and store them under the folder `llmfoundry/scripts/train/evals/`
