"""Training, validation and test entry point.

A run is fully specified by a task (`experiment=`) and a model (`model=`):

    python training_script.py experiment=induction_head model=dg seed=1

See README.md, Section "Reproducing the paper tables".
"""

import logging
import sys
from pathlib import Path
from typing import Literal

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from datasets.factory import get_data_loaders
from models.factory import construct_model
from utils.directories import ROOT_DIR
from utils.environment import get_job_id, get_torch_device
from utils.SeedManager import SeedManager

log = logging.getLogger("script")

BestCriteria = Literal["acc", "loss"]

# Register the `get_job_id` as a custom OmegaConf resolver. This allows the
# output of the function to be accessed when building the Hydra configuration.
OmegaConf.register_new_resolver(name="slurm_job_id", resolver=get_job_id)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):

    seed_manager = SeedManager()
    seed_manager.seed_dataset(cfg.data_seed)
    seed_manager.seed_transformer(cfg.transformer_seed)

    base_dir = Path(HydraConfig.get().runtime.output_dir)
    log.info(f"Results Directory: {base_dir}")

    device = get_torch_device()

    train_loader, val_loader, test_loader = get_data_loaders(cfg=cfg)

    training(
        cfg=cfg,
        base_dir=base_dir,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
    )
    test(cfg=cfg, base_dir=base_dir, device=device, test_loader=test_loader)

def _exact_match_accuracy(predictions: torch.Tensor, labels: torch.Tensor, batch_size: int) -> float:
    # --- EXACT MATCH ACCURACY ---
    # 1. Check which individual tokens match
    token_matches = (predictions == labels)
    # 2. Collapse the sequence dimension: True ONLY if all tokens in the row match
    sequence_matches = torch.all(token_matches, dim=1)
    # 3. Count how many sequences were perfectly predicted
    correct_sequences = torch.sum(sequence_matches).item()
    # 4. Divide by the batch size to get the true accuracy percentage
    accuracy = correct_sequences / batch_size

    return accuracy

def _classification_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return torch.sum(preds == labels).item() / labels.size(0)

def training(
    cfg: DictConfig,
    base_dir: Path,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
):
    log.info(f"In this experiment we have '{cfg.model.description}' trained on '{cfg.dataset.description}'")

    # ------------------------------------PREPARE DIRECTORIES------------------------------
    checkpoints_dir = base_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    # ------------------------------------LOAD DATASET AND MODEL----------------------------
    model_mode = cfg.get("model_mode", "new")
    
    match model_mode:
        case "new":
            log.info("Initializing model from scratch.")
            model = construct_model(cfg=cfg, device=device)
            
        case "load":
            pretrained_path = ROOT_DIR / cfg.get("pretrained_path")
            if not pretrained_path:
                raise ValueError("`pretrained_path` must be provided in config when `model_mode` is 'load'.")
                
            log.info(f"Warm-starting model using weights from: {pretrained_path}")
            model = construct_model(cfg=cfg, device=device)
            
            # Load the checkpoint from the model trained WITHOUT bias
            checkpoint = torch.load(pretrained_path, map_location=device, weights_only=False)
            
            # strict=False allows PyTorch to load the existing weights while leaving the new,
            # uninitialized gate bias untouched.
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            log.info(f"Loaded weights. Missing keys (expected for new bias): {missing_keys}")

            # Retrieve the parameter dictionary of the newly instantiated model
            model_params = dict(model.named_parameters())
            
            # Initialize to zero the new parameters (bias) missing in the checkpoint
            zeroed_keys = []
            with torch.no_grad():
                for key in missing_keys:
                    if "bias" in key and key in model_params:
                        torch.nn.init.zeros_(model_params[key])
                        zeroed_keys.append(key)
            
            log.info(f"Loaded weights. Missing keys (new biases initialized to 0): {zeroed_keys}")
            
            if unexpected_keys:
                log.warning(f"Unexpected keys found in checkpoint: {unexpected_keys}")
                
        case _:
            raise ValueError(f"Unknown model_mode: {model_mode}")


    log.info(f"In this experiment we have '{cfg.model.description}' trained on '{cfg.dataset.description}'")

    # ------------------------------------PREPARE DIRECTORIES------------------------------
    checkpoints_dir = base_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    # ------------------------------------LOAD DATASET AND MODEL----------------------------
    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"Number of parameters: {total_params}")

    # ------------------------------LOSS, OPTIMIZER AND USEFUL VARIABLES--------------------
    loss_function = loss = torch.nn.CrossEntropyLoss(reduction="none")

    opt = AdamW(
        model.parameters(), 
        lr=cfg.lr, 
        weight_decay=getattr(cfg, "weight_decay", 0.01),
        betas=(getattr(cfg, "beta1", 0.9), getattr(cfg, "beta2", 0.999))
    )
    warmup_steps = getattr(cfg, "warmup_steps", len(train_loader)/100)
    
    scheduler = ReduceLROnPlateau(
        opt, 
        mode='min', 
        factor=getattr(cfg, "lr_factor", 0.5),
        patience=getattr(cfg, "patience", 5), 
        min_lr=1e-6
    )

    # Keep track of the best models found so far
    best_val_accuracy = 0.0
    best_val_loss = float("inf")

    # ----------------------------TRAINING, VALIDATION PROCEDURE-----------------------------
    for epoch in range(1, cfg.epochs + 1):
        log.info(f"Epoch {epoch: 3d} / {cfg.epochs}")

        model.train(True)
        cumulative_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader, start=1):

            inputs, labels = inputs.to(device), labels.to(device)

            global_step = (epoch - 1) * len(train_loader) + i
            if global_step <= warmup_steps:
                lr = cfg.lr * (global_step / warmup_steps)
                for param_group in opt.param_groups:
                    param_group['lr'] = lr

            opt.zero_grad()

            _, output = model(inputs)

            loss = loss_function(output, labels)
            loss = loss.mean()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=getattr(cfg, "max_grad_norm", 10.0))
            
            opt.step()

            cumulative_loss += loss.item()

            if not (i - 1) % cfg.debug_every:
                current_lr = opt.param_groups[0]['lr']
                log.info(f"i: {i: 6d} | Loss: {loss:.5f} | LR: {current_lr:.6f}")
                '''
                for name, param in model.named_parameters():
                    param_norm = param.data.norm(2).item()
                    param_grad_norm = param.grad.data.norm(2).item()
                    wandb_logger.log({f"{name}_norm": param_norm})
                    wandb_logger.log({f"{name}_grad_norm": param_grad_norm})
                '''

        log.info(f"Training Average Loss: {(cumulative_loss / len(train_loader)):.5f}")

        # ------------------------VALIDATION PHASE-----------------------------------------
        model.eval()

        with torch.no_grad():
            # Here we are always creating a iterable object over the validation loader, the next()
            # always return a different set of images because shuffle=True in the DataLoader
            val_inputs, val_labels = next(iter(val_loader))
            val_inputs = val_inputs.to(device)
            val_labels = val_labels.to(device)

            out_val_pred, out_val_logits = model(val_inputs)
            val_loss = loss_function(out_val_logits, val_labels)
            val_loss = val_loss.mean()

            if cfg.dataset.get("task_type", "sequence") == "classification":
                val_accuracy = _classification_accuracy(out_val_logits, val_labels)
            else:
                val_accuracy = _exact_match_accuracy(out_val_pred, val_labels, cfg.validation_batch_size)


        log.info(f"Validation Loss: {val_loss:.5f}, Validation Accuracy: {val_accuracy:.3f}")

        if global_step > warmup_steps:
            scheduler.step(val_loss)

        # Save checkpoint
        model_path = checkpoints_dir / f"model_{epoch}_{val_loss:.5f}_{val_accuracy:.3f}.pt"
        torch.save(
            obj={
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "loss": val_loss,
                "accuracy": val_accuracy,
            },
            f=model_path,
        )

        # Save best model in terms of accuracy (override the previous one)
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_best_model(
                model,
                criteria="acc",
                base_dir=base_dir,
                opt=opt,
                loss=val_loss,
                accuracy=val_accuracy,
            )

        # Save best model in terms of loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_best_model(
                model,
                criteria="loss",
                base_dir=base_dir,
                opt=opt,
                loss=val_loss,
                accuracy=val_accuracy,
            )

    # --- EARLY STOPPING ---
        if val_accuracy >= 1.0:
            log.info("Validation accuracy reached 1.00! Triggering early stopping.")
            break

    log.info("Training complete.")
    #wandb_logger.finish()


def save_best_model(
    model: nn.Module,
    criteria: BestCriteria,
    base_dir: Path,
    opt: Optimizer,
    loss: float,
    accuracy: float,
):
    """Save the best model so far, based on the `criteria`."""
    for path in base_dir.glob(f"best_model_{criteria}*.pt"):
        log.debug(f"Removing previous best: {path}")
        path.unlink()

    model_path = base_dir / f"best_model_{criteria}_{loss:.3f}_{accuracy:.3f}.pt"
    torch.save(
        obj={
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
        },
        f=model_path,
    )


def test(cfg: DictConfig, base_dir: Path, device: torch.device, test_loader: DataLoader):
    log.info("Testing best models (found during training) in terms of loss and accuracy")

    # --------------------------INSTANTIATE MODELS---------------------------
    model_acc = construct_model(cfg=cfg, device=device)
    model_loss = construct_model(cfg=cfg, device=device)

    total_params = sum(p.numel() for p in model_acc.parameters())
    log.info(f"Number of parameters: {total_params}")

    # ---------------------UPLOADING PARAMETERS OF THE BEST MODELS---------------
    acc_file = list(base_dir.glob("best_model_acc*.pt"))
    loss_file = list(base_dir.glob("best_model_loss*.pt"))

    if acc_file:
        best_acc = torch.load(f=acc_file[0], weights_only=False)
        log.info(f"Uploaded: {acc_file[0]}")
    else:
        log.error("Best accuracy model not found")
        sys.exit(1)

    if loss_file:
        best_loss = torch.load(f=loss_file[0], weights_only=False)
        log.info(f"Uploaded: {loss_file[0]}")
    else:
        log.error("Best loss model not found")
        sys.exit(1)

    model_acc.load_state_dict(best_acc["model_state_dict"])
    model_loss.load_state_dict(best_loss["model_state_dict"])

    # -------------------------TESTING----------------------------------------
    loss_function = torch.nn.CrossEntropyLoss(reduction="none")


    model_acc.eval()
    with torch.no_grad():
        test_inputs, test_labels = next(iter(test_loader))
        test_inputs = test_inputs.to(device)
        test_labels = test_labels.to(device)

        out_test_pred, out_test_logits = model_acc(test_inputs)
        test_loss = loss_function(out_test_logits, test_labels)
        test_loss = test_loss.mean()

        if cfg.dataset.get("task_type", "sequence") == "classification":
            test_accuracy = _classification_accuracy(out_test_logits, test_labels)
        else:
            test_accuracy = _exact_match_accuracy(out_test_pred, test_labels, cfg.validation_batch_size)

    log.info(f"Best Model (by Accuracy) - Loss: {test_loss:.5f}, Accuracy: {test_accuracy:.3f}")

    # Test phase of best model in terms of loss
    model_loss.eval()

    with torch.no_grad():
        test_inputs, test_labels = next(iter(test_loader))
        test_inputs = test_inputs.to(device)
        test_labels = test_labels.to(device)

        out_test_pred, out_test_logits = model_loss(test_inputs)
        test_loss = loss_function(out_test_logits, test_labels)
        test_loss = test_loss.mean()

        if cfg.dataset.get("task_type", "sequence") == "classification":
            test_accuracy = _classification_accuracy(out_test_logits, test_labels)
        else:
            test_accuracy = _exact_match_accuracy(out_test_pred, test_labels, cfg.validation_batch_size)

    log.info(f"Best Model (by Loss) - Loss: {test_loss:.5f}, Accuracy: {test_accuracy:.3f}")


if __name__ == "__main__":
    main()