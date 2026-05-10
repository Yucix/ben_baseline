import torch
import numpy as np


class AveragePrecisionMeter(object):
    """
    计算多标签任务的指标，与论文对齐:
    - Precision, Recall, F1 (宏平均，对应论文 Table 1 的 Precision, Recall, F1)
    - Micro F1 (对应论文 Table 1 的 Micro-F1)
    """

    def __init__(self, difficult_examples=False):
        self.difficult_examples = difficult_examples
        self.reset()

    def reset(self):
        self.scores = torch.FloatTensor()
        self.targets = torch.LongTensor()

    @staticmethod
    def _average_precision_binary(scores, labels):
        """Compute AP for one class with ranked predictions."""
        order = np.argsort(-scores)
        labels_sorted = labels[order].astype(np.float64)
        pos_total = labels_sorted.sum()
        if pos_total <= 0:
            return np.nan

        tp_cum = np.cumsum(labels_sorted)
        fp_cum = np.cumsum(1.0 - labels_sorted)
        precision = tp_cum / (tp_cum + fp_cum + 1e-12)
        ap = np.sum(precision * labels_sorted) / (pos_total + 1e-12)
        return ap

    def add(self, output, target):
        """
        output: [N, C] logits
        target: [N, C] {0,1}
        """
        if not torch.is_tensor(output):
            output = torch.tensor(output)
        if not torch.is_tensor(target):
            target = torch.tensor(target)

        # 转到 CPU 存储以节省显存
        output = output.detach().cpu()
        target = target.detach().cpu()

        if self.scores.numel() == 0:
            self.scores = output
            self.targets = target
        else:
            self.scores = torch.cat([self.scores, output], dim=0)
            self.targets = torch.cat([self.targets, target], dim=0)

    def value(self):
        """返回每个类别的 AP"""
        if self.scores.numel() == 0:
            return torch.zeros(1)
        ap = torch.zeros(self.scores.size(1))
        for k in range(self.scores.size(1)):
            ap[k] = self.average_precision(self.scores[:, k], self.targets[:, k])
        return ap

    @staticmethod
    def average_precision(output, target):
        """单类 AP 计算"""
        sorted_scores, indices = torch.sort(output, descending=True)
        target_sorted = target[indices]
        tp_cumsum = torch.cumsum(target_sorted, dim=0)
        total_pos = tp_cumsum[-1].item()

        if total_pos == 0:
            return 0.0

        k_indices = torch.arange(1, len(target) + 1, dtype=torch.float32)
        precision_at_k = tp_cumsum / k_indices
        ap = (precision_at_k * target_sorted).sum() / total_pos
        return ap.item()

    def compute_paper_metrics(self, threshold=0.5):
        """
        计算与论文一致的指标。
        逻辑：
        1. Precision/Recall: 各类 P/R 的算术平均 (对应文中的 Precision/Recall)。
        2. F1: 基于平均后的 Precision 和 Recall 计算调和平均 (对应文中的 F1)。
        3. Micro-F1: 全局 TP/FP/FN 计算。
        4. mAP/AP_per_class: 基于排序结果计算每类 AP，再取均值。
        """
        # Sigmoid 激活并二值化
        probs = torch.sigmoid(self.scores).numpy()
        targets = self.targets.numpy()
        preds = (probs >= threshold).astype(np.float32)

        # --- Per Class Metrics ---
        # axis=0 对样本维求和，得到每个类别的 TP, FP, FN
        tp = np.sum((preds == 1) & (targets == 1), axis=0)
        fp = np.sum((preds == 1) & (targets == 0), axis=0)
        fn = np.sum((preds == 0) & (targets == 1), axis=0)

        # 避免除以 0
        p_class = tp / (tp + fp + 1e-10)
        r_class = tp / (tp + fn + 1e-10)
        f1_class = 2 * p_class * r_class / (p_class + r_class + 1e-10)

        # --- Macro Average (对应论文 Precision, Recall, F1 列) ---
        precision = np.mean(p_class) * 100.0
        recall = np.mean(r_class) * 100.0
        # 论文中的 F1 是基于平均后的 P 和 R 计算的
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        # --- Micro Average (对应论文 Micro-F1) ---
        tp_micro = np.sum(tp)
        fp_micro = np.sum(fp)
        fn_micro = np.sum(fn)

        micro_p = tp_micro / (tp_micro + fp_micro + 1e-10) * 100.0
        micro_r = tp_micro / (tp_micro + fn_micro + 1e-10) * 100.0
        micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r + 1e-10)

        # --- mAP / Per-Class AP ---
        ap_class = np.array([
            self._average_precision_binary(probs[:, i], targets[:, i])
            for i in range(targets.shape[1])
        ], dtype=np.float64)
        map_score = np.nanmean(ap_class) * 100.0 if np.any(~np.isnan(ap_class)) else 0.0
        ap_class = np.nan_to_num(ap_class, nan=0.0) * 100.0

        return {
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Micro_F1": micro_f1,
            "mAP": map_score,
            "Per_Class_AP": ap_class,
            "Per_Class_Precision": p_class * 100.0,
            "Per_Class_Recall": r_class * 100.0,
            "Per_Class_F1": f1_class * 100.0,
        }
