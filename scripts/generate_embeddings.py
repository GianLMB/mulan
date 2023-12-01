"""Generate proteins embeddings with transformers pretrained models. 
Embeddings are stored in PT format."""

import os
from argparse import ArgumentParser
from tqdm import tqdm
import torch

from mulan import utils, constants as C


def get_args():
    parser = ArgumentParser(
        prog="generate_embeddings",
        description=__doc__,
    )
    parser.add_argument(
        "fasta_file",
        type=str,
        help="Path to FASTA file containing sequences to be encoded",
    )
    parser.add_argument(
        "model_name",
        type=str,
        default="ankh_large",
        help=f"PLM name to be loaded. Must be one of {C.PLM_ENCODERS.keys()}",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="./embeddings",
        help="Output directory. Defaults to './embeddings'",
    )
    args = parser.parse_args()
    return args


@torch.inference_mode()
def run(model_name, fasta_file, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = utils.load_pretrained_plm(model_name, device=device)
    dataset = utils.parse_fasta(fasta_file)
    os.makedirs(output_dir, exist_ok=True)
    pbar = tqdm(initial=0, total=len(dataset), colour="red", dynamic_ncols=True, ascii="-#")
    pbar.set_description("Embedding sequences")
    for name, sequence in dataset.items():
        # embed sequence
        embedding = utils.embed_sequence(model, tokenizer, sequence)
        utils.save_embedding(embedding, output_dir, name)
        pbar.update(1)


def main():
    args = get_args()
    run(args.model_name, args.fasta_file, args.output_dir)


if __name__ == "__main__":
    main()
