# **ReNO-based T2I Trainer for Pyramidal Flow Model**



## **Overview**

This trainer script is designed to **fine-tune and align the Text-to-Image (T2I) component** of the Pyramidal Flow Model using **Reward-based Noise Optimization (ReNO)**. ReNO enables the optimization of latent noise inputs to improve prompt alignment and visual quality of generated images, without altering the core model parameters. This approach is especially efficient, as it leverages inference-time adjustments rather than retraining the entire model.



## **Key Features**

* **Reward-based Latent Optimization**: Uses LatentNoiseTrainer to iteratively refine latent noise for each input, optimizing it based on custom reward functions (e.g., prompt relevance, aesthetics).

* **Flexible Distributed Training**: Supports distributed training with Accelerator and Fully Sharded Data Parallel (FSDP) configurations.

* **Adaptive Image-Text Dataset Loading**: Loads and preprocesses datasets based on chosen image resolutions (e.g., 384p, 768p), with multi-aspect ratio support.



## **Usage**



1. **Configure Arguments**:

* Use get_reno_args in the trainer script to define necessary arguments such as batch_size, epochs, lr, and dataset paths.

2. **Run Training**:

* Execute the script with the configured arguments:



```bash
python reno_t2i_trainer.py --anno_file path/to/dataset.jsonl --output_dir ./output --epochs 20 --resolution 384p
```



3. **Checkpointing and Logging**:

* Model checkpoints and ReNO-optimized images are saved to output_dir periodically for easy monitoring and analysis.



## **Code Structure**



* `LatentNoiseTrainer`: Handles latent noise optimization based on reward functions, adapting initial noise to improve prompt alignment.

* `train_one_epoch_with_fsdp`: Conducts standard T2I training, integrating ReNO-optimized latents when generating outputs.

* **Data Loading**: Utilizes ImageTextDataset and create_image_text_dataloaders from the Pyramidal Flow repository for efficient, resolution-specific dataset handling.



