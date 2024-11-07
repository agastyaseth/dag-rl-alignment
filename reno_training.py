import os
import sys
import torch
from pathlib import Path
import logging
from train.train_pyramid_flow import get_args, build_model_runner, auto_resume, build_fsdp_plugin, train_one_epoch_with_fsdp
from reno_trainer import LatentNoiseTrainer
from accelerate import Accelerator
from transformers import logging as transformers_logging
from diffusers import logging as diffusers_logging
from dataset import create_length_grouped_video_text_dataloader
from trainer_misc import create_optimizer, cosine_scheduler, constant_scheduler
from reno.rewards.clip import CLIPLoss 
from dataset.dataloaders import ImageTextDataset, create_image_text_dataloaders
# Initialize Logger
logger = logging.getLogger(__name__)

def get_reno_args():
    """Defines and returns command-line arguments for the ReNO-based T2I training pipeline."""
    import argparse

    parser = argparse.ArgumentParser("ReNO T2I Training Script", add_help=True)
    # Task specific parameters
    parser.add_argument("--task", default="t2i", type=str, choices=["t2i"], help="Training for T2I generation")
    parser.add_argument("--batch_size", default=4, type=int, help="Batch size per device")
    parser.add_argument("--epochs", default=20, type=int, help="Number of epochs")
    parser.add_argument("--print_freq", default=20, type=int)
    parser.add_argument("--save_ckpt_freq", default=5, type=int, help="Frequency of checkpoint saving")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory for saving outputs")
    parser.add_argument("--logging_dir", type=str, default="logs", help="Directory for logging")

    # Model parameters
    parser.add_argument("--model_name", default="pyramid_flux", type=str, help="Model name for T2I")
    parser.add_argument("--model_path", default="", type=str, help="Path to pre-trained model weights")
    parser.add_argument("--model_dtype", default="fp16", type=str, choices=["bf16", "fp16"], help="Model precision")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")

    # FSDP parameters
    parser.add_argument("--use_fsdp", action="store_true")
    parser.add_argument("--fsdp_shard_strategy", default="zero2", type=str, choices=["zero2", "zero3"])

    # Data parameters
    parser.add_argument("--anno_file", type=str, required=True, help="Annotation JSONL file for dataset")
    parser.add_argument("--resolution", default="384p", type=str, choices=["384p", "768p"], help="Image resolution")
    parser.add_argument("--num_workers", default=4, type=int, help="Number of dataloader workers")
    
    # ReNO-specific parameters
    parser.add_argument("--n_iters", default=10, type=int, help="Number of ReNO optimization steps per batch")
    parser.add_argument("--n_inference_steps", default=50, type=int, help="Number of inference steps in ReNO optimization")
    parser.add_argument("--save_all_images", action="store_true", help="Save images at each ReNO iteration")
    parser.add_argument("--grad_clip", default=0.1, type=float, help="Gradient clipping value in ReNO optimization")

    # Optimizer and Scheduler
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--warmup_lr", type=float, default=1e-6, help="Warmup learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-5, help="Minimum learning rate")
    parser.add_argument("--lr_scheduler", type=str, default="constant_with_warmup",
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
                        help="Type of learning rate scheduler")
    parser.add_argument("--warmup_epochs", type=int, default=1, help="Epochs to warmup LR")

    return parser.parse_args()


def load_image_text_data(args):
    """Loads and prepares the image-text dataset for T2I training."""
    logger.info("Loading and preparing image-text dataset...")

    if args.resolution == '384p':
        image_ratios = [1/1, 3/5, 5/3]
        image_sizes = [(512, 512), (384, 640), (640, 384)]
    else:
        image_ratios = [1/1, 3/5, 5/3]
        image_sizes = [(1024, 1024), (768, 1280), (1280, 768)]

    image_text_dataset = ImageTextDataset(
        args.anno_file,
        add_normalize=True,
        ratios=image_ratios,
        sizes=image_sizes,
    )

    dataloader = create_image_text_dataloaders(
        image_text_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        multi_aspect_ratio=True,
        epoch=args.seed,
        sizes=image_sizes,
        use_distributed=True,
        world_size=accelerator.num_processes,
        rank=accelerator.process_index,
    )

    logger.info("Dataset loading completed.")
    return dataloader


def main():
    args = get_reno_args()

    # Setup output directories
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    logging_dir = Path(args.output_dir, args.logging_dir)

    # Initialize Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision=args.model_dtype,
        log_with="tensorboard",
        project_config={'project_dir': args.output_dir, 'logging_dir': logging_dir},
        fsdp_plugin=build_fsdp_plugin(args) if args.use_fsdp else None,
    )

    if accelerator.is_main_process:
        transformers_logging.set_verbosity_warning()
        diffusers_logging.set_verbosity_info()
    else:
        transformers_logging.set_verbosity_error()
        diffusers_logging.set_verbosity_error()

    # Build model
    logger.info("Initializing model...")
    runner = build_model_runner(args)
    device = accelerator.device

    # Initialize ReNO
    reward_losses = [CLIPLoss()] #, YourRewardLoss2()]  # Replace with your reward functions
    reno_trainer = LatentNoiseTrainer(
        reward_losses=reward_losses,
        model=runner.dit,
        n_iters=args.n_iters,
        n_inference_steps=args.n_inference_steps,
        seed=args.seed,
        grad_clip=args.grad_clip,
        save_all_images=args.save_all_images,
        device=device,
    )

    # Load dataset
    dataloader = load_image_text_data(args)

    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(runner.dit.parameters(), lr=args.lr, eps=1e-8)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: (1 - step / args.epochs) if step < args.warmup_epochs else 1.0,
    )

    # Prepare model and optimizer with accelerator
    runner.dit, optimizer = accelerator.prepare(runner.dit, optimizer)

    # Training Loop
    accelerator.wait_for_everyone()
    logger.info("Starting ReNO-based T2I training...")

    global_step = auto_resume(args, accelerator)

    for epoch in range(args.epochs):
        for batch in dataloader:
            prompt = batch['text']
            batch_size = batch['images'].size(0)
            latents = torch.randn(batch_size, runner.dit.config.latent_dim, device=device)

            # Optimize latents using ReNO
            initial_image, best_image_pil, initial_rewards, best_rewards = reno_trainer.train(
                latents=latents,
                prompt=prompt,
                optimizer=optimizer,
                save_dir=args.output_dir,
            )

            # Save best image if specified
            if args.save_all_images:
                best_image_pil.save(os.path.join(args.output_dir, f"best_image_{global_step}.png"))
            logger.info(f"Epoch {epoch}, Step {global_step} - Best Reward Loss: {best_rewards}")

            # Run standard training for the epoch
            train_stats = train_one_epoch_with_fsdp(
                runner,
                None,  # EMA model if used
                accelerator,
                args.model_dtype,
                dataloader,
                optimizer,
                lr_scheduler,
                device,
                epoch,
                args.clip_grad,
                start_steps=global_step,
                args=args,
                print_freq=args.print_freq,
                iters_per_epoch=len(dataloader),
            )

            global_step += 1

        # Save checkpoints
        if epoch % args.save_ckpt_freq == 0:
            save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
            accelerator.save_state(save_path)
            logger.info(f"Checkpoint saved at {save_path}")

    accelerator.end_training()
    logger.info("Training completed.")

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()