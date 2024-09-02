"""Generate proteins embeddings with transformers pretrained models. 
Embeddings are stored in PT format."""

import os
from argparse import ArgumentParser
from tqdm import tqdm
import torch

import mulan


def get_args():
    parser = ArgumentParser(
        prog="plm-embed",
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
        default="ankh",
        help=f"PLM name to be loaded. Must be one of {mulan.get_available_plms()}",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="./embeddings",
        help="Output directory. Defaults to './embeddings'",
    )
    args = parser.parse_args()
    if args.model_name not in mulan.get_available_plms():
        raise ValueError(f"Invalid model name: {args.model_name}")
    return args


@torch.inference_mode()
def run(model_name, fasta_file, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = mulan.load_pretrained_plm(model_name, device=device)
    dataset = mulan.utils.parse_fasta(fasta_file)
    os.makedirs(output_dir, exist_ok=True)
    pbar = tqdm(initial=0, total=len(dataset), colour="red", dynamic_ncols=True, ascii="-#")
    pbar.set_description("Embedding sequences")
    for name, sequence in dataset.items():
        # embed sequence
        embedding = mulan.utils.embed_sequence(model, tokenizer, sequence)
        mulan.utils.save_embedding(embedding, output_dir, name)
        pbar.update(1)


def main():
    args = get_args()
    run(args.model_name, args.fasta_file, args.output_dir)


if __name__ == "__main__":
    main()
