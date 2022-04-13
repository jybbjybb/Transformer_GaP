# PyTorch Implementation of Transformer Grow and Prune
Paper "Effective Model Sparsification by Scheduled Grow-and-Prune Methods", https://arxiv.org/pdf/2106.09857.pdf

This Readme explains how to run the scheduled grow-and-prune for sparse Transformers.



## Setup

The following section lists the requirements in order to start training the Transformer model.



## Quick Start Guide
To train your model using mixed or TF32 precision with Tensor Cores or using FP32, perform the following steps using the default parameters of the Transformer model on the [WMT14 English-German](http://statmt.org/wmt14/translation-task.html#Download) dataset. For the specifics concerning training and inference, see the [Advanced](#advanced) section.

1. Clone the repository 
```
git clone https://github.com/jybbjybb/Transformer_GaP.git
cd Transformer_GaP
```

2. Build Transformer PyTorch NGC  container
```bash
docker build . -t your.repository:transformer
nvidia-docker run -it --rm --ipc=host your.repository:transformer bash
```
If you already have preprocessed data, use:
```bash
nvidia-docker run -it --rm --ipc=host -v <path to your preprocessed data>:/data/wmt14_en_de_joined_dict your.repository:transformer bash
```
If you already have data downloaded, but it has not yet been preprocessed, use:
```bash
nvidia-docker run -it --rm --ipc=host -v <path to your unprocessed data>:/workspace/translation/examples/translation/orig your.repository:transformer bash
```
3. Download and preprocess dataset: Download and preprocess the WMT14 English-German dataset.

```bash 
scripts/run_preprocessing.sh
```
After running this command, data will be downloaded to `/workspace/translation/examples/translation/orig` directory and this data will be processed and put into `/data/wmt14_en_de_joined_dict` directory. 

4. Start the GaP
```
python scripts/run_seq_gap.py --ep-per-step 2 --num-steps 3 --extra-cmd='--no-epoch-checkpoints' --extra-cmd-step0='--no-epoch-checkpoints' --global-workspace results/tmp/ --sparsity 0.8 --config-folder profiles/3_step_forward_gap/0.8_std_naming/ --num-parts 3 --partition-type cyclic
```

5. Test the results
```
bash scripts/run_test.sh <your saved checkpoints>
```

