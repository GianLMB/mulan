# MuLAN: MUtational effects with Light Attention Networks

![mulan abstract](./images/visual_abstract.png)

MuLAN is a deep learning method that leverages transfer learning from fundational protein language models 
and light attention to predict mutational effects in protein complexes.  
Inputs to the model are only the sequences of interacting proteins and (optionally) zero-shot scores for the considered mutations. 
Attention weights extracted from the model can give insights on protein interface regions.


## Quick start

### Installation
As a prerequisite, you must have PyTorch installed to use this repository. If not, it can be installed with conda running the following:
```bash
# PyTorch 2.1.0, CUDA 12.1
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```
For other versions or installation methods, please refer to [PyTorch documentation](https://pytorch.org/get-started/locally/).  
MuLAN and its dependencies can then be installed with
```bash
git clone https://github.com/GianLMB/mulan
cd mulan
pip install .
```
We suggest to do it in a dedicated conda environment.


### Usage
We provide several command line interfaces for quick usage of MuLAN different applications:
- `mulan-predict` for $\Delta \Delta G$ prediction of single and multiple-point mutations;
- `mulan-att` to extract residues weights, related to interface regions;
- `mulan-landscape` to produce a full mutational landscape for a given complex;
- `mulan-train` to re-train the model on a custom dataset or to run cross validation (not added yet);
- `plm-embed` to extract embeddings from protein language models. 
Since the script uses the `transformers` interface, only models that are saved on HuggingFace 🤗 Hub can be loaded.

Information about required inputs and usage examples for each command are provided with the `--help` flag.


## Citation


