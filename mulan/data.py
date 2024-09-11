from typing import Dict, List, NamedTuple, Tuple, Optional

import os
from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data import Dataset, default_collate
from torch.nn.utils.rnn import pad_sequence
import numpy as np

from mulan import utils


class MutatedComplex(NamedTuple):
    sequence_A: str
    sequence_B: str
    mutations: Tuple[str]


class MutatedComplexEmbeds(NamedTuple):
    seq1: torch.Tensor
    seq2: torch.Tensor
    mut_seq1: torch.Tensor
    mut_seq2: torch.Tensor


class MulanDataset(Dataset):
    def __init__(
        self,
        mutated_complexes: List[MutatedComplex],
        wt_sequences: Dict[str, str],
        embeddings_dir: str,
        plm_model_name: str = None,
        scores: List[float] = None,
        zs_scores: List[float] = None,
    ):

        self.sequences = wt_sequences
        self.embeddings_dir = embeddings_dir
        self.mutated_complexes = mutated_complexes
        self.zs_scores = zs_scores
        self.scores = scores
        self._sequences_ids = []
        self._fill_metadata(mutated_complexes)

        # generate embeddings if not provided
        all_ids = set([id_ for ids in self._sequences_ids for id_ in ids])
        provided_embeddings_ids = (
            [os.path.splitext(file)[0] for file in os.listdir(self.embeddings_dir)]
            if os.path.exists(self.embeddings_dir)
            else []
        )
        missing_ids = all_ids - set(provided_embeddings_ids)
        if missing_ids:
            if not plm_model_name:
                raise ValueError(
                    "`plm_model_name` must be provided if embeddings were not pre-computed."
                )
            self._generate_missing_embeddings(plm_model_name, missing_ids)

    def __len__(self):
        return len(self.mutated_complexes)

    def __getitem__(self, index):
        return {
            "data": self.mutated_complexes[index],
            "inputs_embeds": self._load_embeddings(index),
            "zs_scores": (
                torch.tensor(self.zs_scores[index], dtype=torch.float32)
                if self.zs_scores
                else None
            ),
            "labels": (
                torch.tensor(self.scores[index], dtype=torch.float32) if self.scores else None
            ),
        }

    @classmethod
    def from_table(
        cls,
        mutated_complexes_file: str,
        wt_sequences_file: str,
        embeddings_dir: str,
        plm_model_name: str = None,
    ):
        wt_sequences = utils.parse_fasta(wt_sequences_file)
        # parse table file
        data = pd.read_table(mutated_complexes_file, sep=r"\s+", header=None)
        mutated_complexes = [
            MutatedComplex(row[0], row[1], tuple(row[2].split(",")))
            for row in data.itertuples(index=False)
        ]
        scores, zs_scores = None, None
        if len(data.columns) > 3:
            scores = data[3].astype(float).tolist()
        if len(data.columns) > 4:
            zs_scores = data[4].astype(float).tolist()
        return cls(
            mutated_complexes, wt_sequences, embeddings_dir, plm_model_name, scores, zs_scores
        )

    def _fill_metadata(self, mutated_complexes):
        for seq1_label, seq2_label, mutations in mutated_complexes:
            seq1 = self.sequences[seq1_label]
            seq2 = self.sequences[seq2_label]
            mut_seq1, mut_seq2 = utils.parse_mutations(mutations, seq1, seq2)
            mut_seq1_label = (
                f"{seq1_label}_{'-'.join([mut for mut in mutations if mut[1] == 'A'])}"
            )
            mut_seq2_label = (
                f"{seq2_label}_{'-'.join([mut for mut in mutations if mut[1] == 'B'])}"
            )
            self.sequences.update({mut_seq1_label: mut_seq1, mut_seq2_label: mut_seq2})
            self._sequences_ids.append((seq1_label, seq2_label, mut_seq1_label, mut_seq2_label))
        return

    def _generate_missing_embeddings(self, plm_model_name, missing_ids):
        plm_model, plm_tokenizer = utils.load_pretrained_plm(plm_model_name)
        os.makedirs(self.embeddings_dir, exist_ok=True)
        for id_ in tqdm(missing_ids, desc="Generating embeddings"):
            seq = self.sequences[id_]
            embedding = utils.embed_sequence(plm_model, plm_tokenizer, seq)
            utils.save_embedding(embedding, self.embeddings_dir, id_)
        # del plm_model, plm_tokenizer
        return

    def _load_embeddings(self, index):
        return MutatedComplexEmbeds(
            *[
                torch.load(os.path.join(self.embeddings_dir, f"{id_}.pt"), weights_only=True)
                for id_ in self._sequences_ids[index]
            ]
        )


class MulanDataCollator(object):
    def __init__(self, padding_value: float = 0.0):
        self.padding_value = padding_value

    def __call__(self, batch):
        return self._collate_fn(batch)

    def _collate_fn(self, batch):
        elem = batch[0]
        if isinstance(elem, dict):
            return {key: self._collate_fn([d[key] for d in batch]) for key in elem}
        if isinstance(elem, MutatedComplexEmbeds):
            return MutatedComplexEmbeds(
                *[
                    pad_sequence(embeds, batch_first=True, padding_value=self.padding_value)
                    for embeds in (zip(*batch))
                ]
            )
        elif elem is None:
            return None
        else:
            return default_collate(batch)


def split_data(
    mutated_complexes_file: str,
    output_dir: Optional[str] = None,
    add_validation_set: bool = True,
    validation_size: float = 0.15,
    test_size: float = 0.15,
    num_folds: int = 1,
    random_state: int = 42,
):
    """Split data into train, validation and test sets for training or cross-validation."""

    def _save_data(data, output_file):
        data.to_csv(output_file, sep="\t", index=False, header=False)

    train_data_all, test_data_all = [], []
    val_data_all = [] if add_validation_set else None
    files_basename = os.path.splitext(os.path.basename(mutated_complexes_file))[0]
    data = pd.read_table(mutated_complexes_file, sep=r"\s+", header=None)
    rng = np.random.default_rng(random_state)
    if num_folds <= 0:
        raise ValueError("`num_folds` must be greater than 0.")
    elif num_folds == 2 and add_validation_set:
        raise ValueError("`num_folds` must be greater than 2 to add a validation set.")
    elif num_folds == 1:
        split_index = rng.choice(
            [0, 1, 2],
            size=len(data),
            p=[test_size, validation_size, 1 - test_size - validation_size],
        )
        test_data = data[split_index == 0]
        if add_validation_set:
            val_data = data[split_index == 1]
            train_data = data[split_index == 2]
            val_data_all.append(val_data)
        else:
            train_data = data[(split_index == 1) & (data[split_index == 2])]
        train_data_all.append(train_data)
        test_data_all.append(test_data)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            _save_data(train_data, os.path.join(output_dir, f"{files_basename}_train.tsv"))
            _save_data(test_data, os.path.join(output_dir, f"{files_basename}_test.tsv"))
            if add_validation_set:
                _save_data(val_data, os.path.join(output_dir, f"{files_basename}_val.tsv"))
    else:
        fold_index = rng.integers(low=0, high=num_folds, size=len(data))
        for test_fold_index in range(num_folds):
            test_data = data[fold_index == test_fold_index]
            if add_validation_set:
                val_fold_index = (test_fold_index - 1) % num_folds
                val_data = data[fold_index == val_fold_index]
                val_data_all.append(val_data)
                train_data = data[(fold_index != test_fold_index) & (fold_index != val_fold_index)]
            else:
                train_data = data[fold_index != test_fold_index]
            train_data_all.append(train_data)
            test_data_all.append(test_data)
            if output_dir:
                os.makedirs(os.path.join(output_dir, f"fold_{test_fold_index}"), exist_ok=True)
                _save_data(
                    train_data,
                    os.path.join(
                        output_dir, f"fold_{test_fold_index}", f"{files_basename}_train.tsv"
                    ),
                )
                _save_data(
                    test_data,
                    os.path.join(
                        output_dir, f"fold_{test_fold_index}", f"{files_basename}_test.tsv"
                    ),
                )
                if add_validation_set:
                    _save_data(
                        val_data,
                        os.path.join(
                            output_dir, f"fold_{test_fold_index}", f"{files_basename}_val.tsv"
                        ),
                    )
    return train_data_all, test_data_all, val_data_all
