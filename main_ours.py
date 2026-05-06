import math
import os
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from torch_geometric.loader import DataLoader

from dataset import EdgeDataset, load_dataset, split_edge
from model_ours import ContextBranch, EdgeClassifier, GAEBranch
from param_parser import parse_args
from train import (
    choose_checkpoint_by_metric,
    collect_probabilities,
    evaluate,
    train_epoch,
    tune_inference_for_macro_f1,
)
from utils_ours import PseudoLabelManager, build_class_weights_from_counts, edge_emb, process_edge, seed_everything

warnings.filterwarnings("ignore", category=UserWarning)


@torch.no_grad()
def export_tsne_features(
    data,
    args,
    encoders,
    classifiers,
    edge_idxs,
    prop_edge_index,
    target_edge_index,
    run_idx,
):
    gae_enc, ctx_enc = encoders
    gae_cls, ctx_cls = classifiers
    for model in [gae_enc, ctx_enc, gae_cls, ctx_cls]:
        model.eval()

    args.training = False

    if args.tsne_split == "all":
        selected_edge_idx = torch.cat([edge_idxs[key] for key in ["train", "val", "test"]], dim=0)
    else:
        selected_edge_idx = edge_idxs[args.tsne_split]

    selected_edge_idx = selected_edge_idx.view(-1).long()
    if args.tsne_max_edges > 0 and len(selected_edge_idx) > args.tsne_max_edges:
        generator = torch.Generator()
        generator.manual_seed(args.seed + run_idx)
        perm = torch.randperm(len(selected_edge_idx), generator=generator)[: args.tsne_max_edges]
        selected_edge_idx = selected_edge_idx[perm]

    selected_edge_idx = selected_edge_idx.to(args.device)

    z_gae = gae_enc.encode(prop_edge_index, args)
    z_ctx = ctx_enc.encode_context(args)
    edge_feat_gae = edge_emb(z_gae, target_edge_index, selected_edge_idx)
    edge_feat_ctx = edge_emb(z_ctx, target_edge_index, selected_edge_idx)
    edge_attr = data.edge_attr[selected_edge_idx]
    hidden_gae = gae_cls.extract_hidden(edge_feat_gae, edge_attr)
    hidden_ctx = ctx_cls.extract_hidden(edge_feat_ctx, edge_attr)
    edge_labels = data.y[selected_edge_idx]

    save_dir = Path(args.tsne_dir) / args.dataset / f"run_{run_idx}" / args.tsne_split
    save_dir.mkdir(parents=True, exist_ok=True)

    np.save(save_dir / f"{args.tsne_split}_gae.npy", hidden_gae.detach().cpu().numpy())
    np.save(save_dir / f"{args.tsne_split}_ctx.npy", hidden_ctx.detach().cpu().numpy())
    np.save(save_dir / "edge_labels.npy", edge_labels.detach().cpu().numpy())
    np.save(save_dir / "edge_indices.npy", selected_edge_idx.detach().cpu().numpy())

    print(f"Saved t-SNE features to {save_dir}")


def run(
    data,
    loaders,
    args,
    encoders,
    classifiers,
    optimizer,
    pseudo_manager,
    unlabeled_edge_idx,
    prop_edge_index,
    target_edge_index,
    class_prior=None,
):
    best_val_bacc = -math.inf
    best_val_f1 = -math.inf
    best_epoch = 0
    patience = 20
    patience_counter = 0

    warmup_epochs = getattr(args, "warmup_epochs", 20)
    gae_enc, ctx_enc = encoders
    gae_cls, ctx_cls = classifiers

    os.makedirs(f"./model/{args.dataset}", exist_ok=True)

    for epoch in range(args.epochs):
        if epoch == warmup_epochs:
            print(f"Warm-up Phase Ended at Epoch {epoch}!")

        train_loss = train_epoch(
            data,
            args,
            encoders,
            classifiers,
            loaders["train"],
            optimizer,
            pseudo_manager,
            unlabeled_edge_idx,
            prop_edge_index,
            target_edge_index,
            current_epoch=epoch,
        )

        if epoch % 5 == 0:
            val_bacc, val_f1_macro = evaluate(
                data,
                args,
                encoders,
                classifiers,
                loaders["val"],
                prop_edge_index,
                target_edge_index,
            )

            if epoch < warmup_epochs:
                if choose_checkpoint_by_metric(val_bacc, val_f1_macro, best_val_bacc, best_val_f1, args):
                    best_val_bacc = val_bacc
                    best_val_f1 = val_f1_macro
                    best_epoch = epoch
                    torch.save(gae_enc.state_dict(), f"./model/{args.dataset}/best_gae_enc.pt")
                    torch.save(ctx_enc.state_dict(), f"./model/{args.dataset}/best_ctx_enc.pt")
                    torch.save(gae_cls.state_dict(), f"./model/{args.dataset}/best_gae_cls.pt")
                    torch.save(ctx_cls.state_dict(), f"./model/{args.dataset}/best_ctx_cls.pt")
            else:
                if choose_checkpoint_by_metric(val_bacc, val_f1_macro, best_val_bacc, best_val_f1, args):
                    args.lambda_u = min(args.lambda_u + 0.1, 2.0)
                    best_val_bacc = val_bacc
                    best_val_f1 = val_f1_macro
                    best_epoch = epoch
                    patience_counter = 0

                    torch.save(gae_enc.state_dict(), f"./model/{args.dataset}/best_gae_enc.pt")
                    torch.save(ctx_enc.state_dict(), f"./model/{args.dataset}/best_ctx_enc.pt")
                    torch.save(gae_cls.state_dict(), f"./model/{args.dataset}/best_gae_cls.pt")
                    torch.save(ctx_cls.state_dict(), f"./model/{args.dataset}/best_ctx_cls.pt")
                else:
                    args.lambda_u = max(args.lambda_u - 0.05, 0.0)
                    patience_counter += 1

                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch}, best epoch {best_epoch}")
                        break

    gae_enc.load_state_dict(torch.load(f"./model/{args.dataset}/best_gae_enc.pt"))
    ctx_enc.load_state_dict(torch.load(f"./model/{args.dataset}/best_ctx_enc.pt"))
    gae_cls.load_state_dict(torch.load(f"./model/{args.dataset}/best_gae_cls.pt"))
    ctx_cls.load_state_dict(torch.load(f"./model/{args.dataset}/best_ctx_cls.pt"))

    inference_cfg = None
    class_prior_cpu = class_prior.detach().cpu() if class_prior is not None else None
    if class_prior_cpu is not None and getattr(args, "tune_inference_for_macro_f1", 1):
        val_prob_gae, val_prob_ctx, val_labels = collect_probabilities(
            data,
            args,
            encoders,
            classifiers,
            loaders["val"],
            prop_edge_index,
            target_edge_index,
        )
        inference_cfg, tuned_val_bacc, tuned_val_f1 = tune_inference_for_macro_f1(
            val_prob_gae,
            val_prob_ctx,
            val_labels,
            class_prior_cpu,
            args,
        )
        print(
            f"Inference tuning | weight={inference_cfg['ensemble_weight']:.2f} | "
            f"prior_tau={inference_cfg['prior_tau']:.2f} | "
            f"Val BACC={tuned_val_bacc:.4f} | Val Macro-F1={tuned_val_f1:.4f}"
        )

    test_bacc, test_f1_macro = evaluate(
        data,
        args,
        encoders,
        classifiers,
        loaders["test"],
        prop_edge_index,
        target_edge_index,
        inference_cfg=inference_cfg,
        class_prior=class_prior_cpu,
    )
    return test_bacc, test_f1_macro


if __name__ == "__main__":
    start = time.time()
    args = parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.path = os.getcwd()
    args.lambda_u = 0.0
    seed_everything(args.seed)
    encoder = SentenceTransformer("all-MiniLM-L6-v2", device=args.device)

    data = load_dataset(args, encoder)
    edge_idxs = split_edge(data, args)
    res = []

    train_edge0 = data.edge_index[:, edge_idxs["train"]].to(args.device)
    train_edge = process_edge(train_edge0).to(args.device)

    unlabeled_edge_idx = edge_idxs.get("unlabeled", torch.tensor([], dtype=torch.long)).to(args.device)
    num_classes = int(torch.unique(data.y).shape[0])
    train_y = data.y[edge_idxs["train"]].view(-1).long().cpu()
    labeled_class_counts = torch.bincount(train_y, minlength=num_classes).float()
    class_prior = (labeled_class_counts / labeled_class_counts.sum()).to(args.device)

    args.reweight = build_class_weights_from_counts(
        labeled_class_counts,
        power=getattr(args, "pseudo_balance_power", 0.5),
        normalize=True,
    ).to(args.device)

    split_data = {key: EdgeDataset(edge_idxs[key]) for key in edge_idxs if key in ["train", "val", "test"]}
    loaders = {}
    for key in split_data:
        loaders[key] = DataLoader(
            split_data[key],
            batch_size=args.batch_size,
            shuffle=(key == "train"),
            collate_fn=split_data[key].collate_fn,
        )

    n_edge_attr = data.edge_attr.shape[1]

    for run_idx in range(args.runs):
        data = data.to(args.device)

        gae_enc = GAEBranch(
            data.num_nodes,
            args.n_embed,
            args.n_hidden,
            backbone=args.model,
            cheb_k=getattr(args, "cheb_k", 2),
        ).to(args.device)
        ctx_enc = ContextBranch(data.num_nodes, args.n_embed, args.n_hidden).to(args.device)
        gae_cls = EdgeClassifier(args.n_hidden, n_edge_attr, num_classes, dropout=args.dropout).to(args.device)
        ctx_cls = EdgeClassifier(args.n_hidden, n_edge_attr, num_classes, dropout=args.dropout).to(args.device)

        encoders = [gae_enc, ctx_enc]
        classifiers = [gae_cls, ctx_cls]

        all_params = (
            list(gae_enc.parameters())
            + list(ctx_enc.parameters())
            + list(gae_cls.parameters())
            + list(ctx_cls.parameters())
        )
        optimizer = torch.optim.Adam(all_params, lr=args.lr, weight_decay=args.wd)

        pseudo_manager = PseudoLabelManager(
            num_unlabeled_edges=data.edge_index.size(1),
            mi_threshold=getattr(args, "mi_threshold", 0.2),
            history_threshold=getattr(args, "history_threshold", 1),
            num_classes=num_classes,
            max_per_class_ratio=getattr(args, "max_per_class_ratio", 3.0),
            labeled_class_counts=labeled_class_counts,
            balance_power=getattr(args, "pseudo_balance_power", 0.5),
            minority_mi_boost=getattr(args, "minority_mi_boost", 0.5),
            min_pseudo_per_class=getattr(args, "min_pseudo_per_class", 1),
        )

        test_bacc, test_f1_macro = run(
            data=data,
            loaders=loaders,
            args=args,
            encoders=encoders,
            classifiers=classifiers,
            optimizer=optimizer,
            pseudo_manager=pseudo_manager,
            unlabeled_edge_idx=unlabeled_edge_idx,
            prop_edge_index=train_edge,
            target_edge_index=data.edge_index,
            class_prior=class_prior,
        )

        res.append({"bacc": test_bacc, "f1_macro": test_f1_macro})

        if args.save_tsne:
            export_tsne_features(
                data=data,
                args=args,
                encoders=encoders,
                classifiers=classifiers,
                edge_idxs=edge_idxs,
                prop_edge_index=train_edge,
                target_edge_index=data.edge_index,
                run_idx=run_idx,
            )

        data = data.to("cpu")

    metrics = ["bacc", "f1_macro"]
    means = {}
    stds = {}
    end = time.time()

    for metric in metrics:
        values = [r[metric] for r in res]
        means[metric] = np.mean(values)
        stds[metric] = np.std(values)

    metrics_output = [f"{metric}: {means[metric]:.3f} ± {stds[metric]:.3f}" for metric in metrics]
    print(f"Total time: {end - start:.2f} seconds, " + ", ".join(metrics_output))
