"""
train.py
Script de entrenamiento para CPU local usando CPUTrainerConfig.

Uso:
    python train.py
    python train.py --val-split 0.15 --seed 42

Flags: --csv, --images-dir, --val-split (default 0.2), --seed (default 0)
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

    # image_size=224 para ser compatible con CPUTrainerConfig (efficientnet_b0)
    preprocess_224 = PreprocessConfig(image_size=224)

    train_ds = RetinaDataset(
        config=DatasetConfig(augment=True, preprocess=preprocess_224),
        csv_path=train_csv,
        images_dir=args.images_dir,
    )
    val_ds = RetinaDataset(
        config=DatasetConfig(augment=False, preprocess=preprocess_224),
        csv_path=val_csv,
        images_dir=args.images_dir,
    )

    trainer = make_trainer("cpu")
    model   = make_model(trainer.config.model_type)

    print(f"Dispositivo : {trainer.config.device}")
    print(f"Modelo      : {trainer.config.model_type}")
    print(f"Image size  : {trainer.config.image_size}")
    print(f"Batch size  : {trainer.config.batch_size}")
    print(f"Épocas      : phase1={trainer.config.phase1_epochs} + phase2={trainer.config.phase2_epochs}")
    print(f"Patience    : {trainer.config.patience}\n")

    history = trainer.fit(model, train_ds, val_ds)

    print(f"\nMejor QWK de validación: {history['best_val_kappa']:.4f}")
    print(f"Checkpoint guardado en : checkpoints/best_model.pt")


if __name__ == "__main__":
    main()
