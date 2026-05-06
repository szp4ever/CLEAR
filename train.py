import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score

from utils_ours import edge_emb, mixup_edge_features


def choose_checkpoint_by_metric(val_bacc, val_macro_f1, best_bacc, best_macro_f1, args):
    metric = getattr(args, "model_select_metric", "macro_f1")
    alpha = getattr(args, "joint_metric_alpha", 0.7)
    eps = 1e-8

    if metric == "bacc":
        return (val_bacc > best_bacc + eps) or (
            abs(val_bacc - best_bacc) <= eps and val_macro_f1 > best_macro_f1 + eps
        )
    if metric == "joint":
        current = alpha * val_macro_f1 + (1 - alpha) * val_bacc
        best = alpha * best_macro_f1 + (1 - alpha) * best_bacc
        return (current > best + eps) or (
            abs(current - best) <= eps and val_macro_f1 > best_macro_f1 + eps
        )
    return (val_macro_f1 > best_macro_f1 + eps) or (
        abs(val_macro_f1 - best_macro_f1) <= eps and val_bacc > best_bacc + eps
    )


def combine_branch_probabilities(prob_gae, prob_ctx, ensemble_weight=0.5):
    return ensemble_weight * prob_gae + (1.0 - ensemble_weight) * prob_ctx


def apply_prior_adjustment(probs, class_prior=None, prior_tau=0.0):
    if class_prior is None or prior_tau <= 0:
        return probs

    adjusted_logits = torch.log(probs.clamp_min(1e-12)) - prior_tau * torch.log(class_prior.clamp_min(1e-12))
    return F.softmax(adjusted_logits, dim=-1)


def predict_with_config(prob_gae, prob_ctx, class_prior=None, inference_cfg=None):
    if inference_cfg is None:
        inference_cfg = {}
    ensemble_weight = inference_cfg.get("ensemble_weight", 0.5)
    prior_tau = inference_cfg.get("prior_tau", 0.0)

    ensemble_prob = combine_branch_probabilities(prob_gae, prob_ctx, ensemble_weight=ensemble_weight)
    adjusted_prob = apply_prior_adjustment(ensemble_prob, class_prior=class_prior, prior_tau=prior_tau)
    return adjusted_prob.argmax(dim=-1)


def compute_metrics_from_predictions(labels, preds):
    labels_np = labels.cpu().numpy()
    preds_np = preds.cpu().numpy()
    bacc = balanced_accuracy_score(labels_np, preds_np)
    macro_f1 = f1_score(labels_np, preds_np, average="macro")
    return bacc, macro_f1


def tune_inference_for_macro_f1(prob_gae, prob_ctx, labels, class_prior, args):
    best_cfg = {"ensemble_weight": 0.5, "prior_tau": 0.0}
    best_bacc, best_macro_f1 = compute_metrics_from_predictions(
        labels, predict_with_config(prob_gae, prob_ctx, class_prior=class_prior, inference_cfg=best_cfg)
    )

    if not getattr(args, "tune_inference_for_macro_f1", 1):
        return best_cfg, best_bacc, best_macro_f1

    max_tau = getattr(args, "prior_adjust_max_tau", 2.0)
    tau_grid = torch.linspace(0.0, max_tau, steps=9).tolist()
    ensemble_grid = [0.3, 0.4, 0.5, 0.6, 0.7]

    for ensemble_weight in ensemble_grid:
        for prior_tau in tau_grid:
            cfg = {"ensemble_weight": ensemble_weight, "prior_tau": float(prior_tau)}
            preds = predict_with_config(prob_gae, prob_ctx, class_prior=class_prior, inference_cfg=cfg)
            bacc, macro_f1 = compute_metrics_from_predictions(labels, preds)
            if macro_f1 > best_macro_f1 + 1e-8 or (
                abs(macro_f1 - best_macro_f1) <= 1e-8 and bacc > best_bacc + 1e-8
            ):
                best_cfg = cfg
                best_bacc = bacc
                best_macro_f1 = macro_f1

    return best_cfg, best_bacc, best_macro_f1


@torch.no_grad()
def collect_probabilities(data, args, encoders, classifiers, eval_loader, prop_edge_index, target_edge_index):
    gae_enc, ctx_enc = encoders
    gae_cls, ctx_cls = classifiers

    args.training = False
    for model in [gae_enc, ctx_enc, gae_cls, ctx_cls]:
        model.eval()

    all_prob_gae = []
    all_prob_ctx = []
    all_labels = []

    z_gae = gae_enc.encode(prop_edge_index, args)
    z_ctx = ctx_enc.encode_context(args)

    for batch_edge_idx, _ in eval_loader:
        batch_edge_idx = batch_edge_idx.to(args.device)

        edge_feat_gae = edge_emb(z_gae, target_edge_index, batch_edge_idx)
        edge_feat_ctx = edge_emb(z_ctx, target_edge_index, batch_edge_idx)
        edge_attr = data.edge_attr[batch_edge_idx]

        logits_gae = gae_cls(edge_feat_gae, edge_attr)
        logits_ctx = ctx_cls(edge_feat_ctx, edge_attr)

        all_prob_gae.append(F.softmax(logits_gae, dim=-1).cpu())
        all_prob_ctx.append(F.softmax(logits_ctx, dim=-1).cpu())
        all_labels.append(data.y[batch_edge_idx].cpu())

    return torch.cat(all_prob_gae), torch.cat(all_prob_ctx), torch.cat(all_labels)


def train_epoch(
    data,
    args,
    encoders,
    classifiers,
    train_loader,
    optimizer,
    pseudo_manager,
    unlabeled_edge_idx,
    prop_edge_index,
    target_edge_index,
    current_epoch,
):
    gae_enc, ctx_enc = encoders
    gae_cls, ctx_cls = classifiers
    warmup_epochs = getattr(args, "warmup_epochs", 20)
    is_warmup = current_epoch < warmup_epochs
    pseudo_loss_weight = pseudo_manager.get_loss_weights(args.device)

    args.training = True
    for model in [gae_enc, ctx_enc, gae_cls, ctx_cls]:
        model.train()

    total_loss = 0.0
    stats = {
        "ctx_to_gae_total": 0,
        "ctx_to_gae_correct": 0,
        "gae_to_ctx_total": 0,
        "gae_to_ctx_correct": 0,
    }
    ctx_to_gae_all_preds = []
    gae_to_ctx_all_preds = []

    for _, (batch_edge_idx, _) in enumerate(train_loader):
        optimizer.zero_grad()

        if isinstance(batch_edge_idx, (list, tuple)):
            if isinstance(batch_edge_idx[0], torch.Tensor):
                batch_edge_idx = torch.stack(batch_edge_idx)
            else:
                batch_edge_idx = torch.tensor(batch_edge_idx, dtype=torch.long)

        batch_edge_idx = batch_edge_idx.to(args.device)

        z_gae = gae_enc.encode(prop_edge_index, args)
        z_ctx = ctx_enc.encode_context(args)

        edge_feat_gae_sup = edge_emb(z_gae, target_edge_index, batch_edge_idx)
        edge_feat_ctx_sup = edge_emb(z_ctx, target_edge_index, batch_edge_idx)
        edge_attr_sup = data.edge_attr[batch_edge_idx]

        logits_gae_sup = gae_cls(edge_feat_gae_sup, edge_attr_sup)
        logits_ctx_sup = ctx_cls(edge_feat_ctx_sup, edge_attr_sup)

        weight = args.reweight if hasattr(args, "reweight") else None
        loss_sup = F.cross_entropy(logits_gae_sup, data.y[batch_edge_idx], weight=weight) + F.cross_entropy(
            logits_ctx_sup, data.y[batch_edge_idx], weight=weight
        )

        if is_warmup:
            loss_sup.backward()
            optimizer.step()
            total_loss += loss_sup.item()
            continue

        batch_size = len(batch_edge_idx)
        if len(unlabeled_edge_idx) > 0:
            perm = torch.randperm(len(unlabeled_edge_idx), device=args.device)[:batch_size]
            current_unlabeled_idx = unlabeled_edge_idx[perm]
        else:
            current_unlabeled_idx = torch.tensor([], dtype=torch.long, device=args.device)

        valid_mask_gae, valid_mask_ctx = None, None
        pl_gae, pl_ctx = None, None
        mean_prob_gae, mean_prob_ctx = None, None
        if len(current_unlabeled_idx) > 0:
            pl_gae, mask_gae, mean_prob_gae = pseudo_manager.generate_pseudo_labels(
                gae_enc,
                gae_cls,
                args,
                prop_edge_index,
                target_edge_index,
                current_unlabeled_idx,
                edge_attr=data.edge_attr,
            )
            pl_ctx, mask_ctx, mean_prob_ctx = pseudo_manager.generate_pseudo_labels(
                ctx_enc,
                ctx_cls,
                args,
                prop_edge_index,
                target_edge_index,
                current_unlabeled_idx,
                edge_attr=data.edge_attr,
            )

            valid_mask_gae, valid_mask_ctx = pseudo_manager.get_valid_masks(
                mask_gae,
                mask_ctx,
                current_unlabeled_idx,
                pseudo_labels_gae=pl_gae,
                pseudo_labels_ctx=pl_ctx,
                mean_prob_gae=mean_prob_gae,
                mean_prob_ctx=mean_prob_ctx,
            )

            if valid_mask_gae.any():
                valid_mask_gae = pseudo_manager.apply_class_balance(
                    valid_mask_gae,
                    current_unlabeled_idx,
                    pl_gae,
                    mean_prob_gae,
                    branch_name="gae",
                )
            if valid_mask_ctx.any():
                valid_mask_ctx = pseudo_manager.apply_class_balance(
                    valid_mask_ctx,
                    current_unlabeled_idx,
                    pl_ctx,
                    mean_prob_ctx,
                    branch_name="ctx",
                )

        loss_unsup_gae = 0
        loss_unsup_ctx = 0

        if len(current_unlabeled_idx) > 0:
            valid_unlabeled_idx_ctx = current_unlabeled_idx[valid_mask_ctx]
            if len(valid_unlabeled_idx_ctx) > 0:
                edge_feat_gae_unsup = edge_emb(z_gae, target_edge_index, valid_unlabeled_idx_ctx)
                edge_attr_unsup_ctx = data.edge_attr[valid_unlabeled_idx_ctx]
                valid_pseudo_y_ctx = pl_ctx[valid_mask_ctx]
                true_y_ctx = data.y[valid_unlabeled_idx_ctx]
                stats["ctx_to_gae_total"] += len(valid_unlabeled_idx_ctx)
                stats["ctx_to_gae_correct"] += (valid_pseudo_y_ctx == true_y_ctx).sum().item()
                ctx_to_gae_all_preds.append(valid_pseudo_y_ctx.detach().cpu())

                mixed_feat_gae, y_a, y_b, lam = mixup_edge_features(edge_feat_gae_unsup, valid_pseudo_y_ctx)
                logits_gae_unsup = gae_cls(mixed_feat_gae, edge_attr_unsup_ctx)
                loss_unsup_gae = lam * F.cross_entropy(logits_gae_unsup, y_a, weight=pseudo_loss_weight) + (
                    1 - lam
                ) * F.cross_entropy(logits_gae_unsup, y_b, weight=pseudo_loss_weight)

            valid_unlabeled_idx_gae = current_unlabeled_idx[valid_mask_gae]
            if len(valid_unlabeled_idx_gae) > 0:
                edge_feat_ctx_unsup = edge_emb(z_ctx, target_edge_index, valid_unlabeled_idx_gae)
                edge_attr_unsup_gae = data.edge_attr[valid_unlabeled_idx_gae]
                valid_pseudo_y_gae = pl_gae[valid_mask_gae]
                true_y_gae = data.y[valid_unlabeled_idx_gae]
                stats["gae_to_ctx_total"] += len(valid_unlabeled_idx_gae)
                stats["gae_to_ctx_correct"] += (valid_pseudo_y_gae == true_y_gae).sum().item()
                gae_to_ctx_all_preds.append(valid_pseudo_y_gae.detach().cpu())

                logits_ctx_unsup = ctx_cls(edge_feat_ctx_unsup, edge_attr_unsup_gae)
                loss_unsup_ctx = F.cross_entropy(
                    logits_ctx_unsup,
                    valid_pseudo_y_gae,
                    weight=pseudo_loss_weight,
                )

        batch_loss = loss_sup + args.lambda_u * (loss_unsup_gae + loss_unsup_ctx)
        batch_loss.backward()
        optimizer.step()
        total_loss += batch_loss.item()

    if not is_warmup:
        ctx_total = stats["ctx_to_gae_total"]
        gae_total = stats["gae_to_ctx_total"]
        ctx_purity = (stats["ctx_to_gae_correct"] / ctx_total * 100) if ctx_total > 0 else 0.0
        gae_purity = (stats["gae_to_ctx_correct"] / gae_total * 100) if gae_total > 0 else 0.0

        ctx_dist_str = "None"
        if len(ctx_to_gae_all_preds) > 0:
            ctx_all = torch.cat(ctx_to_gae_all_preds)
            classes, counts = torch.unique(ctx_all, return_counts=True)
            ctx_dist_str = ", ".join([f"C{int(k)}:{int(v)}" for k, v in zip(classes, counts)])

        gae_dist_str = "None"
        if len(gae_to_ctx_all_preds) > 0:
            gae_all = torch.cat(gae_to_ctx_all_preds)
            classes, counts = torch.unique(gae_all, return_counts=True)
            gae_dist_str = ", ".join([f"C{int(k)}:{int(v)}" for k, v in zip(classes, counts)])

        print("   ↳ [Pseudo-Label Quality & Distribution] ")
        print(f"      CTX -> GAE: Yield: {ctx_total:4d} | Purity: {ctx_purity:5.1f}% | Dist: [{ctx_dist_str}]")
        print(f"      GAE -> CTX: Yield: {gae_total:4d} | Purity: {gae_purity:5.1f}% | Dist: [{gae_dist_str}]")

    return total_loss / len(train_loader)


@torch.no_grad()
def evaluate(
    data,
    args,
    encoders,
    classifiers,
    eval_loader,
    prop_edge_index,
    target_edge_index,
    inference_cfg=None,
    class_prior=None,
):
    prob_gae, prob_ctx, labels = collect_probabilities(
        data,
        args,
        encoders,
        classifiers,
        eval_loader,
        prop_edge_index,
        target_edge_index,
    )
    preds = predict_with_config(prob_gae, prob_ctx, class_prior=class_prior, inference_cfg=inference_cfg)
    return compute_metrics_from_predictions(labels, preds)
