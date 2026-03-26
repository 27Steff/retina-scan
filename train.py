"""
train.py
Script de entrenamiento con soporte para CPU y Apple Silicon (MPS).

Uso:
    python train.py                         # CPU, B0, 224px
    python train.py --preset mps            # Apple Silicon, B4, 380px (~4-8h)
    python train.py --preset mps --val-split 0.15
"""

import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from dataset import RetinaDataset, DatasetConfig
from preprocessing import PreprocessConfig
from model import make_model
from trainer import make_trainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",        default="data/train.csv")
    p.add_argument("--images-dir", default="data/train_images")
    p.add_argument("--val-split",  type=float, default=0.2)
    p.add_argument("--seed",       type=int,   default=0)
    p.add_argument("--preset",     default="cpu", choices=["cpu", "mps", "fast", "standard"],
                   help="Trainer preset: cpu (B0/224px), mps (B4/380px on Apple Silicon)")
    p.add_argument("--model",      default=None,
                   help="Override model backbone, e.g. efficientnet_b4")
    p.add_argument("--image-size", type=int, default=None,
                   help="Override image size")
    return p.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.csv)
    train_df, val_df = train_test_split(
        df,
        test_size=args.val_split,
        stratify=df["diagnosis"],
        random_state=args.seed,
    )

    train_csv = Path("data/split_train.csv")
    val_csv   = Path("data/split_val.csv")
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv,     index=False)

    print(f"Train: {len(train_df)} imágenes | Val: {len(val_df)} imágenes")
    print(f"Distribución train:\n{train_df['diagnosis'].value_counts().sort_index().to_string()}\n")

    trainer = make_trainer(args.preset)

    # Preset defaults por modelo
    _model_defaults = {
        "cpu": ("efficientnet_b0", 224),
        "mps": ("efficientnet_b4", 380),
        "fast": ("efficientnet_b0", 224),
        "standard": ("efficientnet_b4", 380),
    }
    default_model, default_size = _model_defaults[args.preset]
    model_type = args.model or default_model
    image_size = args.image_size or default_size

    preprocess_cfg = PreprocessConfig(image_size=image_size)
    train_ds = RetinaDataset(
        config=DatasetConfig(augment=True, preprocess=preprocess_cfg),
        csv_path=train_csv,
        images_dir=args.images_dir,
    )
    val_ds = RetinaDataset(
        config=DatasetConfig(augment=False, preprocess=preprocess_cfg),
        csv_path=val_csv,
        images_dir=args.images_dir,
    )

    model = make_model(model_type)

    print(f"Preset      : {args.preset}")
    print(f"Dispositivo : {trainer.config.device}")
    print(f"Modelo      : {model_type}")
    print(f"Image size  : {image_size}")
    print(f"Batch size  : {trainer.config.batch_size}")
    print(f"Épocas      : phase1={trainer.config.phase1_epochs} + phase2={trainer.config.phase2_epochs}")
    print(f"Patience    : {trainer.config.patience}\n")

    history = trainer.fit(model, train_ds, val_ds)

    print(f"\nMejor QWK de validación: {history['best_val_kappa']:.4f}")
    print(f"Checkpoint guardado en : checkpoints/best_model.pt")


if __name__ == "__main__":
    main()
