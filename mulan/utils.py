"""Util functions to process data and models."""

from typing import Dict, Iterator, List, Optional, Tuple, Union

import os
import re
import torch
import numpy as np
from scipy.stats import rankdata
from transformers import PreTrainedTokenizerBase, PreTrainedModel

import mulan.constants as C
from mulan.constants import AAs, aa2idx, idx2aa, one2three, three2one


TorchDevice = Union[str, torch.device]


def mutation_generator(sequence: str) -> Iterator[Tuple[str, str]]:
    """Generate all possible single-point mutations for a given sequence.
    Args:
        sequence (str): The input sequence for which mutations are generated.
    Yields:
        A tuple containing the mutation identifier and the mutated sequence.
    """
    for i, aa in enumerate(sequence):
        for new_aa in C.AAs:
            if new_aa != aa:
                yield (f"{aa}{i + 1}{new_aa}", sequence[:i] + new_aa + sequence[i + 1 :])


def parse_mutations(mutations: Tuple[str], seq1: str, seq2: str) -> Tuple[str, str]:
    """
    Parses a list of mutations and applies them to two sequences.
    Args:
        mutations (Tuple[str]): A tuple of strings representing the mutations.
            Each mutation should be in the format '<wt_aa><chain:A,B><position><mut_aa>'.
            Example: 'AB23G'.
        seq1 (str): The first sequence.
        seq2 (str): The second sequence.
    Returns:
        Tuple[str, str]: A tuple representing the modified sequences.
    """
    seq1, seq2 = list(seq1), list(seq2)
    for single_mut in mutations:
        chain = single_mut[1]
        if chain == "A":
            seq1[int(single_mut[2:-1]) - 1] = single_mut[-1]
        else:
            seq2[int(single_mut[2:-1]) - 1] = single_mut[-1]
    return "".join(seq1), "".join(seq2)


def alphabetic_tokens_permutation(tokenizer: PreTrainedTokenizerBase) -> List[int]:
    """Get indices of standard amino acids in alphabetic order for input tokenizer."""
    vocab = tokenizer.get_vocab()
    aas_idx = [vocab[tok] for tok in C.AAs]
    return aas_idx


def parse_fasta(fasta_file: str) -> Dict[str, str]:
    """Parse a fasta file and return a dictionary."""
    with open(fasta_file) as f:
        lines = f.readlines()
    fasta_dict = {}
    for line in lines:
        if line.startswith("#") or not line.strip():
            continue
        if line.startswith(">"):
            key = line.strip().split()[0][1:]
            key = key.split("|")[0]
            fasta_dict[key] = ""
        else:
            fasta_dict[key] += line.strip().upper()
    return fasta_dict


def dict_to_fasta(fasta_dict: Dict[str, str], fasta_file: str):
    """Write a dictionary to a fasta file."""
    with open(fasta_file, "w") as f:
        for key, value in fasta_dict.items():
            f.write(f">{key}\n")
            f.write(f"{value}\n")


def get_available_plms() -> List[str]:
    """Return the names of the available pretrained language models."""
    return list(C.PLM_ENCODERS.keys())


def get_available_models() -> List[str]:
    """Return the names of the available Mulan models."""
    return list(C.MODELS.keys())


def load_pretrained_plm(model_name: str, device: Optional[TorchDevice] = None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = C.PLM_ENCODERS.get(model_name)
    if model_id is None:
        raise ValueError(
            f"Invalid model_name: {model_name}. Must be one of {get_available_plms()}"
        )

    if "t5" in model_id or "ankh" in model_id:
        from transformers import T5EncoderModel

        model = T5EncoderModel.from_pretrained(model_id)
        if "ankh" in model_id:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_id)
        else:
            from transformers import T5Tokenizer

            tokenizer = T5Tokenizer.from_pretrained(model_id)
    else:
        try:
            from transformers import AutoTokenizer, AutoModel

            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModel.from_pretrained(model_id)
        except Exception as e:
            raise e
    model = model.to(device)
    model = model.eval()
    return model, tokenizer


def load_pretrained(pretrained_model_name: str, device: TorchDevice = None, **kwargs):
    """Load a pretrained model from disk."""
    model_path = C.MODELS.get(pretrained_model_name)
    if model_path is None:
        raise ValueError(f"Invalid model_name: {pretrained_model_name}")
    from mulan.modules import LightAttModel

    return LightAttModel.from_pretrained(model_path, device=device, **kwargs)


@torch.inference_mode()
def embed_sequence(
    plm_model: PreTrainedModel, plm_tokenizer: PreTrainedTokenizerBase, sequence: str
):
    """Embed a sequence using a pretrained model."""
    sequence = sequence.upper()
    sequence = re.sub(r"[UZOB]", "X", sequence)  # always replace non-canonical AAs with X
    # Pre-process sequence for ProtTrans models
    if "Rostlab/prot" in plm_tokenizer.name_or_path:
        sequence = " ".join(sequence)
    inputs = plm_tokenizer(
        sequence,
        return_tensors="pt",
        add_special_tokens=True,
        return_special_tokens_mask=True,
    ).to(plm_model.device)
    embedding = (
        plm_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        .last_hidden_state[~inputs["special_tokens_mask"].bool()]
        .unsqueeze(0)
    )
    return embedding


def save_embedding(embedding: torch.Tensor, output_dir: str, name: str):
    """Save an embedding to disk."""
    embedding = embedding.squeeze(0).cpu()
    torch.save(embedding, os.path.join(output_dir, name + ".pt"))


def ranksort(array: np.ndarray) -> np.ndarray:
    """Ranksort an array."""
    return (rankdata(array) / array.size).reshape(array.shape)


def minmax_scale(array: np.ndarray) -> np.ndarray:
    """Min-max scale an array."""
    return (array - array.min()) / (array.max() - array.min())
