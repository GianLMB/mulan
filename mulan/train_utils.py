from typing import Any, Dict, List, Mapping, Optional, Tuple, Union
from dataclasses import dataclass, field
import torch
import torch.nn as nn
from transformers import Trainer
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from scipy.stats import pearsonr, spearmanr

from mulan.data import MutatedComplexEmbeds, MutatedComplex


def _metric_spearmanr(y_true, y_pred):
    return spearmanr(y_true, y_pred, nan_policy="omit")[0]


def _metric_pearsonr(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]


_DEFAULT_METRICS = {
    "mae": mean_absolute_error,
    "rmse": root_mean_squared_error,
    "pcc": _metric_pearsonr,
    "scc": _metric_spearmanr,
}


def default_compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.flatten()
    labels = labels.flatten()
    res = {}
    for name, metric in _DEFAULT_METRICS.items():
        res[name] = metric(labels, predictions)
    return res


@dataclass
class DatasetArguments:
    train_data: str = field(
        metadata={
            "help": (
                "Training data in TSV format. Must contain columns for sequence_A, sequence_B,"
                " mutations (separated with comma if multiple), score and (optionally) zero-shot"
                " score."
            )
        }
    )
    train_fasta_file: str = field(
        metadata={
            "help": (
                "Fasta file containing wild-type sequences for training data. Identifiers must"
                " match the training data and, if present, the evaluation data."
            )
        }
    )
    embeddings_dir: str = field(
        metadata={
            "help": (
                "Directory containing pre-computed embeddings in PT format, or where new"
                " embeddings will be stored. In the latter case, `plm_model_name` must be"
                " provided."
            )
        }
    )
    eval_data: Optional[str] = field(
        default=None,
        metadata={"help": "Evaluation data file, with the same format of training data."},
    )
    test_data: Optional[str] = field(
        default=None,
        metadata={"help": "Test data file, with the same format of training data."},
    )
    test_fasta_file: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Fasta file containing wild-type sequences. Identifiers must match the test data."
            )
        },
    )
    plm_model_name: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Name of the pre-trained protein language model to use for embedding generation."
            )
        },
    )


@dataclass
class ModelArguments:
    model_name_or_config_path: str = field(
        metadata={
            "help": (
                "Name of the pre-trained model to fine-tune, or path to config file in JSON"
                " format."
            )
        }
    )
    save_model: bool = field(
        default=False,
        metadata={"help": "Whether to save the model after training."},
    )


@dataclass
class CustomisableTrainingArguments:
    output_dir: str = field(metadata={"help": "Directory where the trained model will be saved."})
    num_epochs: int = field(default=30, metadata={"help": "Number of training epochs."})
    batch_size: int = field(default=8, metadata={"help": "Batch size."})
    learning_rate: float = field(default=5e-4, metadata={"help": "Learning rate."})
    disable_tqdm: bool = field(
        default=False, metadata={"help": "Whether to disable tqdm progress bars."}
    )
    report_to: Union[None, str, List[str]] = field(
        default="none",
        metadata={"help": "The list of integrations to report the results and logs to."},
    )
    early_stopping_patience: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of epochs without improvement before early stopping. If not set, early"
                " stopping is disabled."
            )
        },
    )


class MulanTrainer(Trainer):
    """Custom Trainer class adapted for Mulan model training"""

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Computes the loss for MulanDataset inputs for a model that do not return loss values.
        """
        inputs.pop("data")
        outputs = model(inputs["inputs_embeds"], inputs.get("zs_scores"))
        labels = inputs.get("labels")
        loss = torch.nn.functional.mse_loss(outputs.view(-1), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

    def _prepare_input(self, data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
        """
        Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        Adapted from the parent class to handle the case where the input is a custom type.
        """
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (MutatedComplexEmbeds, MutatedComplex)):
            return type(data)(*[self._prepare_input(v) for v in data])
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = {"device": self.args.device}
            if self.is_deepspeed_enabled and (
                torch.is_floating_point(data) or torch.is_complex(data)
            ):
                # NLP models inputs are int/uint and those get adjusted to the right dtype of the
                # embedding. Other models such as wav2vec2's inputs are already float and thus
                # may need special handling to match the dtypes of the model
                kwargs.update(
                    {"dtype": self.accelerator.state.deepspeed_plugin.hf_ds_config.dtype()}
                )
            return data.to(**kwargs)
        return data

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Overridden from the parent class to handle the case where the model does not return loss values.
        Support for sagemaker was removed!

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = (
            False
            if len(self.label_names) == 0
            else all(inputs.get(k) is not None for k in self.label_names)
        )
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        # print("return_loss", return_loss, "has_labels", has_labels)
        loss_without_labels = True if len(self.label_names) == 0 and return_loss else False

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels:
            labels = inputs.get("labels")
        else:
            labels = None

        with torch.no_grad():
            if has_labels or loss_without_labels:
                with self.compute_loss_context_manager():
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()
                logits = outputs
            else:
                loss = None
                with self.compute_loss_context_manager():
                    outputs = model(inputs["inputs_embeds"], inputs.get("zs_scores"))
                logits = outputs
                # TODO: this needs to be fixed and made cleaner later.
                if self.args.past_index >= 0:
                    self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        return (loss, logits, labels)
