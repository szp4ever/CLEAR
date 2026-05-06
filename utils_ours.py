import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, to_undirected


def build_class_weights_from_counts(class_counts, power=1.0, normalize=True):
    class_counts = class_counts.float().clamp(min=1.0)
    weights = class_counts.sum() / (class_counts * class_counts.numel())
    weights = weights.pow(power)
    if normalize:
        weights = weights / weights.mean()
    return weights


def edge_emb(x, edge_index, edge_idx):
    head, tail = edge_index[:, edge_idx][0], edge_index[:, edge_idx][1]
    head_emb = x[head]
    tail_emb = x[tail]
    return torch.cat([head_emb, tail_emb], dim=1)


def mixup_edge_features(edge_features, labels, alpha=2.0):
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        lam = 1.0

    batch_size = edge_features.size(0)
    index = torch.randperm(batch_size).to(edge_features.device)
    mixed_edges = lam * edge_features + (1 - lam) * edge_features[index]
    y_a, y_b = labels, labels[index]
    return mixed_edges, y_a, y_b, lam


class PseudoLabelManager:
    def __init__(
        self,
        num_unlabeled_edges,
        mi_threshold,
        history_threshold,
        num_classes=2,
        max_per_class_ratio=2.0,
        labeled_class_counts=None,
        balance_power=0.5,
        minority_mi_boost=0.5,
        min_pseudo_per_class=1,
    ):
        self.mi_threshold = mi_threshold
        self.history_threshold = history_threshold
        self.num_classes = num_classes
        self.max_per_class_ratio = max_per_class_ratio
        self.balance_power = balance_power
        self.minority_mi_boost = minority_mi_boost
        self.min_pseudo_per_class = min_pseudo_per_class

        if labeled_class_counts is None:
            labeled_class_counts = torch.ones(num_classes)
        self.labeled_class_counts = labeled_class_counts.float().clamp(min=1.0)
        self.class_balance_weights = build_class_weights_from_counts(
            self.labeled_class_counts, power=self.balance_power, normalize=True
        )

        self.bayesian_counts_gae = torch.zeros(num_unlabeled_edges)
        self.bayesian_counts_ctx = torch.zeros(num_unlabeled_edges)
        self.selected_class_counts_gae = torch.zeros(num_classes)
        self.selected_class_counts_ctx = torch.zeros(num_classes)

    def calculate_mi(self, prob_samples):
        mean_prob = prob_samples.mean(dim=0)
        entropy_of_mean = -torch.sum(mean_prob * torch.log(mean_prob + 1e-8), dim=-1)
        entropy_samples = -torch.sum(prob_samples * torch.log(prob_samples + 1e-8), dim=-1)
        mean_of_entropy = entropy_samples.mean(dim=0)
        mi = entropy_of_mean - mean_of_entropy
        return mi, mean_prob

    def get_loss_weights(self, device):
        return self.class_balance_weights.to(device)

    def get_class_aware_mi_thresholds(self, device):
        class_weights = self.class_balance_weights.to(device)
        scaling = 1.0 + self.minority_mi_boost * (class_weights - 1.0)
        scaling = scaling.clamp(min=0.25, max=3.0)
        return self.mi_threshold * scaling

    def get_branch_history_weights(self, branch_name, device):
        if branch_name == "gae":
            branch_counts = self.selected_class_counts_gae.to(device)
        elif branch_name == "ctx":
            branch_counts = self.selected_class_counts_ctx.to(device)
        else:
            return torch.ones(self.num_classes, device=device)

        history_weights = (branch_counts.sum() + self.num_classes) / (branch_counts + 1.0)
        history_weights = history_weights / history_weights.mean()
        return history_weights

    def update_selected_class_counts(self, branch_name, selected_labels):
        if selected_labels.numel() == 0:
            return

        selected_counts = torch.bincount(selected_labels, minlength=self.num_classes).float().cpu()
        if branch_name == "gae":
            self.selected_class_counts_gae += selected_counts
        elif branch_name == "ctx":
            self.selected_class_counts_ctx += selected_counts

    def allocate_balanced_targets(self, available_counts, class_weights):
        available_counts = available_counts.long()
        total_available = int(available_counts.sum().item())
        if total_available == 0:
            return torch.zeros_like(available_counts)

        active_mask = available_counts > 0
        active_classes = int(active_mask.sum().item())
        if active_classes == 0:
            return torch.zeros_like(available_counts)

        if active_classes == 1:
            target_budget = max(self.min_pseudo_per_class, int(np.ceil(total_available * 0.5)))
            target_budget = min(target_budget, total_available)
            target_counts = torch.zeros_like(available_counts)
            target_counts[active_mask] = target_budget
            return target_counts

        min_positive = int(available_counts[active_mask].min().item())
        max_per_class = max(
            self.min_pseudo_per_class,
            int(np.ceil(min_positive * self.max_per_class_ratio)),
        )
        capacity = torch.minimum(available_counts, torch.full_like(available_counts, max_per_class))
        target_budget = min(total_available, int(active_classes * max_per_class))

        weighted_counts = available_counts.float() * class_weights
        if weighted_counts.sum() <= 0:
            weighted_counts = available_counts.float()

        raw_targets = weighted_counts / weighted_counts.sum() * target_budget
        target_counts = torch.floor(raw_targets).long()

        for cls in range(self.num_classes):
            if available_counts[cls] > 0:
                target_counts[cls] = max(
                    target_counts[cls].item(),
                    min(self.min_pseudo_per_class, available_counts[cls].item()),
                )

        target_counts = torch.minimum(target_counts, capacity)
        remaining = target_budget - int(target_counts.sum().item())
        remainders = raw_targets - torch.floor(raw_targets)
        spare = capacity - target_counts

        while remaining > 0 and int(spare.max().item()) > 0:
            masked_remainders = remainders.clone()
            masked_remainders[spare <= 0] = -1
            cls = int(masked_remainders.argmax().item())
            if masked_remainders[cls] < 0:
                break
            target_counts[cls] += 1
            spare[cls] -= 1
            remainders[cls] = 0
            remaining -= 1

        return target_counts

    @torch.no_grad()
    def generate_pseudo_labels(
        self,
        encoder,
        classifier,
        args,
        prop_edge_index,
        target_edge_index,
        target_edge_idx,
        edge_attr=None,
        num_samples=10,
    ):
        args.training = True
        encoder.train()
        classifier.train()

        prob_list = []
        for _ in range(num_samples):
            if hasattr(encoder, "encode_context"):
                z = encoder.encode_context(args)
            else:
                z = encoder.encode(prop_edge_index, args)

            edge_features = edge_emb(z, target_edge_index, target_edge_idx)
            if edge_attr is not None:
                edge_attr_batch = edge_attr[target_edge_idx]
                logits = classifier(edge_features, edge_attr_batch)
            else:
                logits = classifier(edge_features)
            prob_list.append(F.softmax(logits, dim=-1))

        prob_samples = torch.stack(prob_list)
        mi, mean_prob = self.calculate_mi(prob_samples)
        _, pseudo_labels = mean_prob.max(dim=-1)

        class_thresholds = self.get_class_aware_mi_thresholds(mean_prob.device)
        mi_mask = mi < class_thresholds[pseudo_labels]

        return pseudo_labels, mi_mask, mean_prob

    def get_valid_masks(
        self,
        mi_mask_gae,
        mi_mask_ctx,
        current_edge_idx,
        pseudo_labels_gae=None,
        pseudo_labels_ctx=None,
        mean_prob_gae=None,
        mean_prob_ctx=None,
    ):
        self.bayesian_counts_gae = self.bayesian_counts_gae.to(current_edge_idx.device)
        self.bayesian_counts_ctx = self.bayesian_counts_ctx.to(current_edge_idx.device)

        self.bayesian_counts_gae[current_edge_idx[mi_mask_gae]] += 1
        self.bayesian_counts_ctx[current_edge_idx[mi_mask_ctx]] += 1

        final_mask_gae = mi_mask_gae & (
            self.bayesian_counts_gae[current_edge_idx] >= self.history_threshold
        )
        final_mask_ctx = mi_mask_ctx & (
            self.bayesian_counts_ctx[current_edge_idx] >= self.history_threshold
        )
        return final_mask_gae, final_mask_ctx

    def apply_class_balance(
        self,
        valid_mask,
        current_edge_idx,
        pseudo_labels,
        mean_prob,
        branch_name=None,
    ):
        if not valid_mask.any():
            return valid_mask

        valid_positions = torch.nonzero(valid_mask, as_tuple=True)[0]
        valid_labels = pseudo_labels[valid_mask]
        valid_probs = mean_prob[valid_mask]
        if valid_probs.dim() > 1:
            valid_probs = valid_probs.max(dim=1)[0]

        class_counts = torch.bincount(valid_labels, minlength=self.num_classes)
        class_weights = self.class_balance_weights.to(valid_labels.device)
        history_weights = self.get_branch_history_weights(branch_name, valid_labels.device)
        effective_weights = class_weights * history_weights
        effective_weights = effective_weights / effective_weights.mean()
        target_counts = self.allocate_balanced_targets(class_counts, effective_weights)

        result_mask = torch.zeros_like(valid_mask)
        selected_labels = []

        for cls in range(self.num_classes):
            cls_positions = valid_positions[valid_labels == cls]
            if cls_positions.numel() == 0:
                continue

            cls_probs = valid_probs[valid_labels == cls]
            n_select = min(cls_positions.numel(), int(target_counts[cls].item()))
            if n_select <= 0:
                continue

            _, top_idx = cls_probs.topk(n_select)
            result_mask[cls_positions[top_idx]] = True
            selected_labels.append(
                torch.full((n_select,), cls, dtype=torch.long, device=valid_labels.device)
            )

        if selected_labels:
            self.update_selected_class_counts(branch_name, torch.cat(selected_labels))

        return result_mask


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def process(edge_index, edge_attr):
    edge_index, edge_attr = to_undirected(
        edge_index,
        edge_attr,
        num_nodes=edge_index.max().item() + 1,
        reduce="mean",
    )
    edge_index, edge_attr = add_self_loops(
        edge_index,
        edge_attr,
        num_nodes=edge_index.max().item() + 1,
        fill_value="mean",
    )
    return edge_index, edge_attr


def process_edge(edge_index):
    edge_index = to_undirected(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=edge_index.max().item() + 1)
    return edge_index


def process_edge_attr(edge_index, edge_feature):
    edge_index, edge_feature = to_undirected(
        edge_index,
        edge_feature,
        num_nodes=edge_index.max().item() + 1,
        reduce="mean",
    )
    return edge_index, edge_feature.type(torch.LongTensor)
