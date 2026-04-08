from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction.text import HashingVectorizer
from torch.utils.data import DataLoader
from torchvision import models, transforms


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ui_scene.data.dataset import load_manifest_records
from ui_scene.data.image_dataset import ImageClassificationDataset
from ui_scene.data.multimodal_dataset import MultimodalClassificationDataset
from ui_scene.models.multimodal_model import MultimodalSceneClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Extract embeddings from a trained checkpoint.')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint.')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--limit', type=int, default=None)
    return parser.parse_args()


def build_eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def build_image_model(backbone_name: str, num_classes: int) -> torch.nn.Module:
    if backbone_name == 'mobilenet_v3_small':
        try:
            model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        except Exception:
            model = models.mobilenet_v3_small(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
        return model

    if backbone_name == 'resnet18':
        try:
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        except Exception:
            model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f'Unsupported backbone: {backbone_name}')


def extract_image_features(model: torch.nn.Module, backbone_name: str, images: torch.Tensor) -> torch.Tensor:
    if backbone_name == 'mobilenet_v3_small':
        features = model.features(images)
        pooled = model.avgpool(features)
        flattened = torch.flatten(pooled, 1)
        return model.classifier[:-1](flattened)

    if backbone_name == 'resnet18':
        features = model.conv1(images)
        features = model.bn1(features)
        features = model.relu(features)
        features = model.maxpool(features)
        features = model.layer1(features)
        features = model.layer2(features)
        features = model.layer3(features)
        features = model.layer4(features)
        features = model.avgpool(features)
        return torch.flatten(features, 1)

    raise ValueError(f'Unsupported backbone: {backbone_name}')


def build_collate_fn(tree_input_dim: int):
    vectorizer = HashingVectorizer(
        n_features=tree_input_dim,
        alternate_sign=False,
        norm='l2',
    )

    def collate_fn(batch: list[dict]) -> dict:
        images = torch.stack([item['image'] for item in batch], dim=0)
        labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
        tree_texts = [item['tree_text'] for item in batch]
        tree_features = vectorizer.transform(tree_texts).toarray()
        return {
            'sample_id': [item['sample_id'] for item in batch],
            'label': labels,
            'label_name': [item['label_name'] for item in batch],
            'image': images,
            'tree_features': torch.tensor(tree_features, dtype=torch.float32),
        }

    return collate_fn


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    config = checkpoint['config']
    label_to_index = checkpoint['label_to_index']
    path_cfg = config['paths']
    train_cfg = config['train']
    model_cfg = config['model']
    task = train_cfg['task']

    manifest_path = Path(path_cfg['manifest_root']) / f'{args.split}.jsonl'
    records = load_manifest_records(manifest_path)
    if args.limit is not None:
        records = records[: args.limit]

    image_size = int(train_cfg['image_size'])
    batch_size = int(args.batch_size)
    num_workers = int(train_cfg.get('num_workers', 0))
    eval_transform = build_eval_transform(image_size)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.backends.cudnn.benchmark = True
    loader_kwargs = {
        'num_workers': num_workers,
        'pin_memory': use_cuda,
    }
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = 2

    device = torch.device('cuda' if use_cuda else 'cpu')
    non_blocking = device.type == 'cuda'
    sample_ids: list[str] = []
    label_names: list[str] = []
    labels: list[int] = []
    embeddings: list[np.ndarray] = []

    if task == 'image_classification':
        dataset = ImageClassificationDataset(records, label_to_index, eval_transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)
        model = build_image_model(model_cfg['image_backbone'], len(label_to_index))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        with torch.no_grad():
            for batch in dataloader:
                images = batch['image'].to(device, non_blocking=non_blocking)
                feats = extract_image_features(model, model_cfg['image_backbone'], images)
                embeddings.append(feats.cpu().numpy())
                sample_ids.extend(batch['sample_id'])
                label_names.extend(batch['label_name'])
                labels.extend(batch['label'].tolist())

    elif task == 'multimodal_classification':
        tree_input_dim = int(train_cfg.get('tree_input_dim', 512))
        max_tree_tokens = int(train_cfg.get('max_tree_tokens', 256))
        tree_profile = str(train_cfg.get('tree_profile', 'legacy'))
        dataset = MultimodalClassificationDataset(
            records,
            label_to_index,
            image_transform=eval_transform,
            max_tree_nodes=max_tree_tokens,
            tree_profile=tree_profile,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=build_collate_fn(tree_input_dim),
            **loader_kwargs,
        )
        model = MultimodalSceneClassifier(
            backbone_name=model_cfg['image_backbone'],
            image_dim=int(model_cfg.get('image_dim', model_cfg.get('fusion_dim', 256))),
            tree_input_dim=tree_input_dim,
            tree_dim=int(model_cfg.get('tree_hidden_dim', 128)),
            fusion_dim=int(model_cfg.get('fusion_dim', 256)),
            num_classes=len(label_to_index),
            fusion_mode=str(model_cfg.get('fusion_mode', 'concat')),
            tree_encoder_type=str(model_cfg.get('tree_encoder_type', 'simple')),
            dropout=float(model_cfg.get('dropout', 0.1)),
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        with torch.no_grad():
            for batch in dataloader:
                images = batch['image'].to(device, non_blocking=non_blocking)
                tree_features = batch['tree_features'].to(device, non_blocking=non_blocking)
                _, fused = model(images, tree_features)
                embeddings.append(fused.cpu().numpy())
                sample_ids.extend(batch['sample_id'])
                label_names.extend(batch['label_name'])
                labels.extend(batch['label'].tolist())

    else:
        raise ValueError(f'Unsupported task in checkpoint config: {task}')

    embedding_array = np.concatenate(embeddings, axis=0) if embeddings else np.empty((0, 0), dtype=np.float32)

    checkpoint_tag = f'{checkpoint_path.parent.name}_{checkpoint_path.stem}'
    output_dir = Path(path_cfg['output_root']) / 'embeddings' / checkpoint_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{args.split}.npz'
    np.savez_compressed(
        output_path,
        embeddings=embedding_array,
        labels=np.array(labels, dtype=np.int64),
        label_names=np.array(label_names, dtype=object),
        sample_ids=np.array(sample_ids, dtype=object),
    )

    meta_path = output_dir / f'{args.split}_meta.json'
    with meta_path.open('w', encoding='utf-8') as fp:
        json.dump(
            {
                'checkpoint': str(checkpoint_path),
                'task': task,
                'split': args.split,
                'num_samples': len(sample_ids),
                'embedding_dim': int(embedding_array.shape[1]) if embedding_array.size else 0,
            },
            fp,
            ensure_ascii=False,
            indent=2,
        )

    print(f'Saved embeddings to: {output_path}')


if __name__ == '__main__':
    main()
