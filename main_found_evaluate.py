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

import argparse
from model import FoundModel
from misc import load_config
from datasets.datasets import build_dataset
from evaluation.saliency import evaluate_saliency
from evaluation.uod import evaluation_unsupervised_object_discovery

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = 'Evaluation of FOUND',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--eval-type",
        type=str,
        choices=["saliency", "uod"],
        help="Evaluation type."
    )
    parser.add_argument(
        "--dataset-eval",
        type=str,
        choices=["ECSSD", "DUT-OMRON", "DUTS-TEST", "VOC07", "VOC12", "COCO20k"],
        help="Name of evaluation dataset."
    )
    parser.add_argument(
        "--dataset-set-eval",
        type=str,
        default=None,
        help="Set of the dataset."
    )
    parser.add_argument(
        "--apply-bilateral",
        action="store_true", 
        help="use bilateral solver."
    )
    parser.add_argument(
        "--evaluation-mode",
        type=str,
        default="multi",
        choices=["single", "multi"],
        help="Type of evaluation."
    )
    parser.add_argument(
        "--model-weights",
        type=str,
        default="data/weights/decoder_weights.pt",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/found_DUTS-TR.yaml",
    )
    args = parser.parse_args()
    print(args.__dict__)

    # Configuration
    config = load_config(args.config)

    # ------------------------------------
    # Load the model
    model = FoundModel(vit_model=config.model["pre_training"],
                        vit_arch=config.model["arch"],
                        vit_patch_size=config.model["patch_size"],
                        enc_type_feats=config.found["feats"],
                        bkg_type_feats=config.found["feats"],
                        bkg_th=config.found["bkg_th"])
    # Load weights
    model.decoder_load_weights(args.model_weights)
    model.eval()
    print(f"Model {args.model_weights} loaded correctly.")

    # ------------------------------------
    # Build the validation set
    val_dataset = build_dataset(
        root_dir=args.dataset_dir,
        dataset_name=args.dataset_eval,
        dataset_set=args.dataset_set_eval,
        for_eval=True,
        evaluation_type=args.eval_type,
    )
    print(f"\nBuilding dataset {val_dataset.name} (#{len(val_dataset)} images)")
    
    # ------------------------------------
    # Training
    print(f"\nStarted evaluation on {val_dataset.name}")
    if args.eval_type == "saliency":
        evaluate_saliency(
            val_dataset,
            model=model,
            evaluation_mode=args.evaluation_mode,
            apply_bilateral=args.apply_bilateral,
        )
    elif args.eval_type == "uod":
        if args.apply_bilateral:
            raise ValueError("Not implemented.")

        evaluation_unsupervised_object_discovery(
            val_dataset,
            model=model,
            evaluation_mode=args.evaluation_mode,
        )
    else:
        raise ValueError("Other evaluation method to come.")