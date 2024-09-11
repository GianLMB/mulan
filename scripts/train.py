"""Train Mulan model on custom data using HuggingFace Trainer API"""

import os
import pandas as pd
import torch
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    logging,
    EarlyStoppingCallback,
    set_seed,
)

import mulan
from mulan.data import MulanDataset, MulanDataCollator
from mulan.train_utils import (
    DatasetArguments,
    ModelArguments,
    CustomisableTrainingArguments,
    MulanTrainer,
    default_compute_metrics,
)


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def get_args():
    parser = HfArgumentParser(
        dataclass_types=[DatasetArguments, ModelArguments, CustomisableTrainingArguments],
        prog="mulan-train",
        description=__doc__,
    )
    data_args, model_args, custom_training_args = parser.parse_args_into_dataclasses()
    return (data_args, model_args, custom_training_args)


def load_data(data_args):
    logger.info("Loading training data")  
    train_dataset = MulanDataset.from_table(
        data_args.train_data,
        data_args.train_fasta_file,
        data_args.embeddings_dir,
        data_args.plm_model_name,
    )
    eval_dataset, test_dataset = None, None
    if data_args.eval_data:
        eval_dataset = MulanDataset.from_table(
            data_args.eval_data,
            data_args.train_fasta_file,
            data_args.embeddings_dir,
            data_args.plm_model_name,
        )
    if data_args.test_data:
        logger.info("Loading test data...")
        test_dataset = MulanDataset.from_table(
            data_args.test_data,
            data_args.test_fasta_file,
            data_args.embeddings_dir,
            data_args.plm_model_name,
        )
    return train_dataset, eval_dataset, test_dataset


def dummy_forward_call(model, dataset, data_collator):
    inputs = data_collator([dataset[0]])
    return model(inputs["inputs_embeds"], inputs.get("zs_scores"))


def save_predictions(output_dir, dataset, predictions):
    df = pd.DataFrame(dataset.mutated_complexes)
    df["mutations"] = df["mutations"].apply(lambda x: ",".join(x))
    df["score"] = predictions
    df.to_csv(
        os.path.join(output_dir, "test_predictions.tsv"), sep="\t", index=False, header=False
    )
    return


def save_model_ckpt(model, output_dir):
    torch.save(
        {"state_dict": model.state_dict(), "config": model.config.__dict__},
        os.path.join(output_dir, "model.ckpt"),
    )


def train(data_args, model_args, custom_training_args):
    # set global seed
    set_seed(42)

    # load data
    train_dataset, eval_dataset, test_dataset = load_data(data_args)

    # load model
    if model_args.model_name_or_config_path in mulan.get_available_models():
        model = mulan.load_pretrained(model_args.model_name_or_config_path, device="cpu")
    else:
        config = mulan.MulanConfig.from_json(model_args.model_name_or_config_path)
        model = mulan.LightAttModel(config)

    # training arguments
    training_args = TrainingArguments(
        output_dir=custom_training_args.output_dir,
        num_train_epochs=custom_training_args.num_epochs,
        per_device_train_batch_size=custom_training_args.batch_size,
        per_device_eval_batch_size=custom_training_args.batch_size,
        logging_dir=custom_training_args.output_dir,
        report_to=custom_training_args.report_to,
        remove_unused_columns=False,
        label_names=["labels"],
        logging_strategy="epoch",
        eval_strategy="epoch" if eval_dataset else "no",
        save_strategy="epoch" if model_args.save_model else "no",
        load_best_model_at_end=(eval_dataset and model_args.save_model),  # to be tested
        metric_for_best_model="loss",
        save_total_limit=2,
    )

    data_collator = MulanDataCollator(padding_value=model.config.padding_value)
    optimizer = torch.optim.AdamW(model.parameters(), lr=custom_training_args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    early_stopping = (
        EarlyStoppingCallback(custom_training_args.early_stopping_patience)
        if custom_training_args.early_stopping_patience
        else None
    )
    dummy_forward_call(model, train_dataset, data_collator)  # to initialize lazy modules

    # instantiate Trainer
    trainer = MulanTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=default_compute_metrics,
        optimizers=(optimizer, scheduler),
        callbacks=[early_stopping] if early_stopping else None,
    )

    # train model
    train_results = trainer.train()
    metrics = train_results.metrics

    if test_dataset:
        prediction_results = trainer.predict(test_dataset)
        save_predictions(
            custom_training_args.output_dir, test_dataset, prediction_results.predictions
        )
        metrics.update(prediction_results.metrics)

    trainer.save_metrics("all", metrics)

    # TODO
    # remove logging message `Trainer.model is not a `PreTrainedModel`, only saving its state dict.``
    if model_args.save_model:
        save_model_ckpt(model, custom_training_args.output_dir)

    return


def main():
    data_args, model_args, custom_training_args = get_args()
    train(data_args, model_args, custom_training_args)


if __name__ == "__main__":
    main()
