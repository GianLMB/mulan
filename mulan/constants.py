"""Constants used in the package."""

import os 


# Single letter, three letter, and full amino acid names.
aa_names = (
    ('A', 'ALA', 'alanine'),
    ('R', 'ARG', 'arginine'),
    ('N', 'ASN', 'asparagine'),
    ('D', 'ASP', 'aspartic acid'),
    ('C', 'CYS', 'cysteine'),
    ('E', 'GLU', 'glutamic acid'),
    ('Q', 'GLN', 'glutamine'),
    ('G', 'GLY', 'glycine'),
    ('H', 'HIS', 'histidine'),
    ('I', 'ILE', 'isoleucine'),
    ('L', 'LEU', 'leucine'),
    ('K', 'LYS', 'lysine'),
    ('M', 'MET', 'methionine'),
    ('F', 'PHE', 'phenylalanine'),
    ('P', 'PRO', 'proline'),
    ('S', 'SER', 'serine'),
    ('T', 'THR', 'threonine'),
    ('W', 'TRP', 'tryptophan'),
    ('Y', 'TYR', 'tyrosine'),
    ('V', 'VAL', 'valine'),
    # Extended AAs
    ('B', 'ASX', 'asparagine or aspartic acid'),
    ('Z', 'GLX', 'glutamine or glutamic acid'),
    ('X', 'XAA', 'Any'),
    ('J', 'XLE', 'Leucine or isoleucine'),
)

# Indices of standard amino acids in `aa_names`.
standard_indices = tuple(range(20))

# Single letter codes of standard amino acids.
standard_aas = tuple(aa_names[i][0] for i in standard_indices)
AAs = tuple(sorted(standard_aas))

# aa_to_idx and idx_to_aa
aa2idx = dict(zip(AAs, standard_indices))
idx2aa = {v: k for k, v in aa2idx.items()}

# dictionaries for aas names conversion
one2three = dict(aa_names[i][:2] for i in standard_indices)
three2one = {v: k for k, v in one2three.items()}


# Models names and paths
_BASE_DIR = os.getcwd()  # os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_MODELS_DIR = os.path.join(_BASE_DIR, "models/pretrained")
MODELS_DIR = os.environ.get("MULAN_MODELS_DIR", _DEFAULT_MODELS_DIR)
MODELS = {
    "mulan-esm": f"{MODELS_DIR}/mulan_esm.ckpt",
    "mulan-esm-multiple": f"{MODELS_DIR}/mulan_esm_multiple.ckpt",
    "imulan-esm": f"{MODELS_DIR}/imulan_esm.ckpt",
    "mulan-ankh": f"{MODELS_DIR}/mulan_ankh.ckpt",
    "imulan-ankh": f"{MODELS_DIR}/imulan_ankh.ckpt",
    "mulan-ankh-multiple": f"{MODELS_DIR}/mulan_ankh_multiple.ckpt",
}

# PLMs encoders and HuggingFace Hub ids
PLM_ENCODERS = {
    "esm": "facebook/esm2_t36_3B_UR50D",
    "ankh": "ElnaggarLab/ankh-large",
    "esm_35M": "facebook/esm2_t12_35M_UR50D",
    "esm_650M": "facebook/esm2_t33_650M_UR50D",
    "ankh_base": "ElnaggarLab/ankh-base",
    "protbert": "Rostlab/prot_bert",
    "prott5_xl_half": "Rostlab/prot_t5_xl_half_uniref50-enc",
}
    