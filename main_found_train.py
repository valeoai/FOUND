# Copyright 2022 - Valeo Comfort and Driving Assistance - Oriane Siméoni @ valeo.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import argparse

import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm

from model import FoundModel
from evaluation.saliency import evaluate_saliency
from misc import (
    batch_apply_bilateral_solver,
    set_seed,
    load_config,
)

from datasets.datasets import build_dataset

def train_model(
    model,
    config,
    dataset,
    dataset_dir,
    visualize_freq=10,
    save_model_freq=500,
    tensorboard_log_dir=None,
):

    # Diverse
    print(f"Data will be saved in {tensorboard_log_dir}")
    save_dir = tensorboard_log_dir
    if tensorboard_log_dir is not None:
        # Logging
        if not os.path.exists(tensorboard_log_dir):
            os.makedirs(tensorboard_log_dir)
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(tensorboard_log_dir)

    # Deconvolution
    sigmoid = nn.Sigmoid()
    model.decoder.train()
    model.decoder.to("cuda")

    # ----------------------
    # Optimization
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
                                  model.decoder.parameters(),
                                  lr=config.training["lr0"]
                                 )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.training["step_lr_size"],
        gamma=config.training["step_lr_gamma"],
    )

    # Dataset   
    trainloader = torch.utils.data.DataLoader(dataset, 
                                            batch_size=config.training["batch_size"], 
                                            shuffle=True, 
                                            num_workers=2
                                            )

    n_iter = 0
    for epoch in range(config.training["nb_epochs"]):
        running_loss = 0.0
        tbar = tqdm(enumerate(trainloader, 0), leave=None)
        for i, data in tbar:

            # get the inputs; data is a list of [inputs, inputs_nonorm, labels, img_paths]
            inputs, input_nonorm, gt_labels, img_path = data
            inputs = inputs.to("cuda")
            gt_labels = gt_labels.to("cuda")

            # zero the parameter gradients
            optimizer.zero_grad()

            # Forward steps
            preds, _, shape_f, att = model.forward_step(inputs)

            # -------------------------------------------
            # Bilateral solver loss
            # Compute mask detection
            preds_mask = (sigmoid(preds.detach()) > 0.5).float()

            # Apply bilateral solver
            preds_mask_bs, _ = batch_apply_bilateral_solver(
                                    data,
                                    preds_mask.detach()
                                )
            
            # Compute loss
            flat_preds = preds.permute(0, 2, 3, 1).reshape(-1, 1)
            preds_bs_loss = config.training["w_bs_loss"] * criterion(
                flat_preds, preds_mask_bs.reshape(-1).float()[:,None]
            )
            writer.add_scalar("Loss/self_bs", preds_bs_loss, n_iter)
            loss = preds_bs_loss

            # -------------------------------------------
            # Apply bkg loss
            if n_iter < config.training["stop_bkg_loss"]:
                
                # Get pseudo_labels used as gt
                masks, _ = model.get_bkg_pseudo_labels_batch(
                            att=att,
                            shape_f=shape_f,
                            data=data,
                            shape=preds.shape[-2:],
                        )

                # pseudo_mask vs preds [loss]
                flat_labels = masks.reshape(-1)
                bkg_loss = criterion(
                    flat_preds, flat_labels.float()[:, None]
                )
                writer.add_scalar("Loss/loss", bkg_loss, n_iter)
                loss += bkg_loss
            
            # Add regularization when bkg loss stopped
            else:
                self_loss = criterion(
                            flat_preds, preds_mask.reshape(-1).float()[:,None]
                        )
                self_loss = config.training["w_self_loss"] * self_loss
                loss += self_loss
                writer.add_scalar("Loss/self_loss", self_loss, n_iter)
            
            # Visualize predictions in tensorboard
            if n_iter % visualize_freq == 0:
                grid = torchvision.utils.make_grid(input_nonorm[:5])
                writer.add_image("training/images", grid, n_iter)
                p_grid = torchvision.utils.make_grid(preds_mask[:5])
                writer.add_image("training/preds", p_grid, n_iter)
                
                # Visualize masks
                if n_iter < config.training["stop_bkg_loss"]:
                    p_grid = torchvision.utils.make_grid(masks[:5].unsqueeze(1))
                    writer.add_image("training/bkg_masks", p_grid, n_iter)

            loss.backward()
            optimizer.step()
            writer.add_scalar("Loss/total_loss", loss, n_iter)
            writer.add_scalar("params/lr", optimizer.param_groups[0]["lr"], n_iter)
            scheduler.step()

            # Statistics
            running_loss += loss.item()
            tbar.set_description(
                f"{dataset.name}| train | iter {n_iter} | loss: ({running_loss / (i + 1):.3f}) "
            )

            # Save model
            if n_iter % save_model_freq == 0 and n_iter > 0:
                model.decoder_save_weights(save_dir, n_iter)

            # Evaluation
            if n_iter % config.evaluation["freq"] == 0 and n_iter > 0:
                for dataset_eval_name in config.evaluation["datasets"]:
                    val_dataset = build_dataset(
                                    root_dir=dataset_dir,
                                    dataset_name=dataset_eval_name,
                                    for_eval=True,
                                    dataset_set=None,
                                )
                    evaluate_saliency(
                        val_dataset,
                        model=model,
                        n_iter=n_iter,
                        writer=writer
                    )
        
            if n_iter == config.training["max_iter"]:
                model.decoder_save_weights(save_dir, n_iter)
                print("\n----"
                      "\nTraining done.")
                writer.close()
                return model

            n_iter += 1
    
    # Save model
    model.decoder_save_weights(save_dir, n_iter)

    writer.close()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                description = 'Training of FOUND',
                formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
    parser.add_argument(
        "--exp-name",
        type=str,
        default=None,
        help="Exp name."
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="outputs",
        help="Logging and output directory."
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Root directories of training and evaluation datasets."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/found_DUTS-TR.yaml",
        help="Path of config file."
    )
    parser.add_argument(
        "--save-model-freq",
        type=int,
        default=250,
        help="Frequency of model saving."
    )
    parser.add_argument(
        "--visualization-freq",
        type=int,
        default=50,
        help="Frequency of prediction visualization in tensorboard."
    )
    

    args = parser.parse_args()
    print(args.__dict__)

    # Configuration
    config = load_config(args.config)

    # Exp name
    exp_name = "FOUND-{}-{}{}".format(
                                    config.training["dataset"],
                                    config.model["arch"],
                                    config.model["patch_size"]
                                )

    if args.exp_name is not None:
        exp_name = f"{args.exp_name}-{exp_name}"
        
    # Log dir
    output_dir = os.path.join(
                    args.log_dir,
                    exp_name
                 )
    # Logging
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save config
    with open(f'{output_dir}/config.json', 'w') as f:
        print(f"Config saved in {output_dir}/config.json.")
        json.dump(args.__dict__, f)
    
    # ------------------------------------
    # Set seed
    set_seed(config.training["seed"])

    # ------------------------------------
    # Build the training set
    dataset = build_dataset(
                root_dir=args.dataset_dir,
                dataset_name=config.training["dataset"],
                dataset_set=config.training["dataset_set"],
                config=config,
                for_eval=False,
            )

    dataset_set = config.training["dataset_set"]
    str_set = dataset_set if dataset_set is not None else ""
    print(f"\nBuilding dataset {dataset.name}{str_set} of {len(dataset)}")

    # ------------------------------------
    # Define the model
    model = FoundModel(
                vit_model=config.model["pre_training"],
                vit_arch=config.model["arch"],
                vit_patch_size=config.model["patch_size"],
                enc_type_feats=config.found["feats"],
                bkg_type_feats=config.found["feats"],
                bkg_th=config.found["bkg_th"]
            )

    # ------------------------------------
    # Training
    print(f"\nStarted training on {dataset.name} [tensorboard dir: {output_dir}]")
    model = train_model(
                model=model,
                config=config,
                dataset=dataset,
                dataset_dir=args.dataset_dir,
                tensorboard_log_dir=output_dir,
                visualize_freq=args.visualization_freq,
                save_model_freq=args.save_model_freq,
            )
    print(f"\nTraining done, FOUND model saved in {output_dir}.")