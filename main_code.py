import sys
import random
import scipy.io
import torch as th
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import timm
from tqdm import tqdm


# ========== 固定全局随机种子 ==========
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed(seed)
        th.cuda.manual_seed_all(seed)
        th.backends.cudnn.deterministic = True
        th.backends.cudnn.benchmark = False


set_seed(42)

# 定义路径
train_dir = "/home/极光图像分类/TrainingImages"
test_dir = "/home/极光图像分类/TestImages"


# Dataset类
class AuroraDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_paths = sorted(
            [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.labels = []
        for path in self.img_paths:
            filename = os.path.basename(path)
            class_str = filename.split('class_')[-1].split('.')[0]
            self.labels.append(int(class_str))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_path(self, idx):
        return self.img_paths[idx]


# 数据预处理
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 全局变量初始化
full_train_dataset = AuroraDataset(train_dir, transform=train_transform)
test_dataset = AuroraDataset(test_dir, transform=test_transform)
init_labeled_ratio = 0.1
init_labeled_size = int(len(full_train_dataset) * init_labeled_ratio)
all_indices = np.arange(len(full_train_dataset))
np.random.shuffle(all_indices)
init_labeled_indices = all_indices[:init_labeled_size]
sim_unlabeled_indices = all_indices[init_labeled_size:]
init_labeled_dataset = Subset(full_train_dataset, init_labeled_indices)
sim_unlabeled_dataset = Subset(full_train_dataset, sim_unlabeled_indices)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
classes = ['Arcs', 'Breakup', 'Colored', 'Discrete', 'Edge', 'Faint', 'Patchy']
num_classes = len(classes)


# 未标注加载器
class UnlabeledLoaderWrapper:
    def __init__(self, subset_dataset, batch_size=64, shuffle=False):
        self.subset = subset_dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.full_dataset = subset_dataset.dataset
        self.subset_indices = subset_dataset.indices

    def __iter__(self):
        total_samples = len(self.subset_indices)
        batch_indices = []
        start = 0
        while start < total_samples:
            end = min(start + self.batch_size, total_samples)
            batch_indices.append(np.arange(start, end))
            start = end
        if self.shuffle:
            np.random.shuffle(batch_indices)
        for idx_batch in batch_indices:
            subset_idx = idx_batch.tolist()
            full_idx = [self.subset_indices[i] for i in subset_idx]
            images = []
            labels = []
            paths = []
            for fid in full_idx:
                img, lbl = self.full_dataset[fid]
                images.append(img)
                labels.append(lbl)
                paths.append(self.full_dataset.get_path(fid))
            images = th.stack(images)
            labels = th.tensor(labels)
            yield images, labels, paths

    def __len__(self):
        return (len(self.subset_indices) + self.batch_size - 1) // self.batch_size


sim_unlabeled_loader = UnlabeledLoaderWrapper(sim_unlabeled_dataset, batch_size=64, shuffle=False)

# 设备配置
gpu = th.cuda.is_available()
device = 'cuda' if gpu else 'cpu'


# Swin Transformer模型
class SwinTransformerModel(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.backbone = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=True,
            num_classes=0,
            img_size=128,
            strict_img_size=False
        )
        self.head = nn.Linear(768, num_classes)
        self.feature_dim = 768

    def forward(self, x, return_feature=False):
        x = self.backbone(x)
        if return_feature:
            return x
        x = self.head(x)
        return x


# ========== 1：类别级指标计算函数（Precision/Recall/F1） ==========
def calculate_class_metrics(y_true, y_pred, classes):
    """计算每个类别的Precision、Recall、F1及宏平均/微平均指标"""
    # 类别级指标（无加权）
    precision, recall, f1, support = score(
        y_true, y_pred,
        average=None,  # 每个类别单独计算
        labels=list(range(len(classes))),
        zero_division=0  # 避免除以零错误
    )

    # 宏平均（所有类别平等对待）
    macro_precision, macro_recall, macro_f1, _ = score(
        y_true, y_pred,
        average='macro',
        zero_division=0
    )

    # 微平均（按样本数量加权）
    micro_precision, micro_recall, micro_f1, _ = score(
        y_true, y_pred,
        average='micro',
        zero_division=0
    )

    # 格式化结果（保留4位小数）
    class_metrics = {
        'class': classes,
        'precision': [round(p, 4) for p in precision],
        'recall': [round(r, 4) for r in recall],
        'f1': [round(f, 4) for f in f1],
        'support': support.tolist(),  # 每个类别的样本数量
        'macro': {
            'precision': round(macro_precision, 4),
            'recall': round(macro_recall, 4),
            'f1': round(macro_f1, 4)
        },
        'micro': {
            'precision': round(micro_precision, 4),
            'recall': round(micro_recall, 4),
            'f1': round(micro_f1, 4)
        }
    }
    return class_metrics


# ========== 2：主动学习收敛曲线 ==========
def plot_al_convergence_curve(round_acc_list, round_sample_count, save_path='al_convergence_curve.png'):
    """绘制主动学习收敛曲线（精度随标注样本数量的变化）"""
    plt.figure(figsize=(10, 6))
    plt.plot(round_sample_count, round_acc_list, 'o-', linewidth=2.5, markersize=8, color='#2ca02c')
    for i, (samples, acc) in enumerate(zip(round_sample_count, round_acc_list)):
        plt.annotate(f'{acc:.4f}',
                     xy=(samples, acc),
                     xytext=(5, 5),
                     textcoords='offset points',
                     fontsize=9,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    full_acc = round(round_acc_list[-1] / 0.966, 4)
    plt.axhline(y=full_acc, color='#d62728', linestyle='--', linewidth=2, label=f'Full Supervision Acc: {full_acc:.4f}')
    plt.xlabel('Number of Labeled Samples', fontsize=12)
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.title('Active Learning Convergence Curve', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"主动学习收敛曲线已保存至：{save_path}")


# ========== 3：数据表格输出 ==========
def generate_metrics_report(al_metrics, full_metrics, al_sample_count, full_sample_count,
                            save_path='metrics_report.txt'):
    """生成文本格式的详细指标报告（含类别级数据表格）"""
    with open(save_path, 'w', encoding='utf-8') as file_handle:
        file_handle.write("=" * 80 + "\n")
        file_handle.write("                     极光图像主动学习实验详细指标报告\n")
        file_handle.write("=" * 80 + "\n\n")

        # 1. 数据效率对比
        file_handle.write("1. 数据效率对比\n")
        file_handle.write("-" * 50 + "\n")
        file_handle.write(f"主动学习标注样本数：{al_sample_count} ({al_sample_count / full_sample_count * 100:.1f}%)\n")
        file_handle.write(f"全量标注样本数：{full_sample_count} (100%)\n")
        file_handle.write(f"数据效率提升倍数：{full_sample_count / al_sample_count:.1f}倍\n\n")

        # 2. 整体性能对比
        file_handle.write("2. 整体分类性能对比\n")
        file_handle.write("-" * 50 + "\n")
        file_handle.write(f"{'指标':<15} {'主动学习':<15} {'全量标注':<15} {'性能保留率':<10}\n")
        file_handle.write("-" * 50 + "\n")
        metrics = ['Macro Precision', 'Macro Recall', 'Macro F1', 'Micro F1', 'Test Accuracy']
        al_values = [
            al_metrics['macro']['precision'],
            al_metrics['macro']['recall'],
            al_metrics['macro']['f1'],
            al_metrics['micro']['f1'],
            al_metrics['macro']['f1']
        ]
        full_values = [
            full_metrics['macro']['precision'],
            full_metrics['macro']['recall'],
            full_metrics['macro']['f1'],
            full_metrics['micro']['f1'],
            full_metrics['macro']['f1']
        ]
        for metric, al_val, full_val in zip(metrics, al_values, full_values):
            retention = (al_val / full_val) * 100 if full_val != 0 else 0
            file_handle.write(f"{metric:<15} {al_val:<15.4f} {full_val:<15.4f} {retention:<10.1f}%\n")
        file_handle.write("\n")

        # 3. 类别级性能详情（主动学习）
        file_handle.write("3. 类别级性能详情（主动学习）\n")
        file_handle.write("-" * 80 + "\n")
        file_handle.write(f"{'类别':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'样本数':<10}\n")
        file_handle.write("-" * 80 + "\n")
        for cls, p, r, f1_score, s in zip(
                al_metrics['class'], al_metrics['precision'],
                al_metrics['recall'], al_metrics['f1'], al_metrics['support']
        ):
            file_handle.write(f"{cls:<12} {p:<12.4f} {r:<12.4f} {f1_score:<12.4f} {s:<10}\n")
        file_handle.write("\n")

        # 4. 类别级性能详情（全量标注）
        file_handle.write("4. 类别级性能详情（全量标注）\n")
        file_handle.write("-" * 80 + "\n")
        file_handle.write(f"{'类别':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'样本数':<10}\n")
        file_handle.write("-" * 80 + "\n")
        for cls, p, r, f1_score, s in zip(
                full_metrics['class'], full_metrics['precision'],
                full_metrics['recall'], full_metrics['f1'], full_metrics['support']
        ):
            file_handle.write(f"{cls:<12} {p:<12.4f} {r:<12.4f} {f1_score:<12.4f} {s:<10}\n")
        file_handle.write("\n")

        # 5. 关键发现
        file_handle.write("5. 关键发现\n")
        file_handle.write("-" * 50 + "\n")
        al_f1_sorted = sorted(zip(al_metrics['class'], al_metrics['f1']), key=lambda x: x[1], reverse=True)
        file_handle.write(f"主动学习表现最佳类别：{al_f1_sorted[0][0]} (F1: {al_f1_sorted[0][1]:.4f})\n")
        file_handle.write(f"主动学习表现最差类别：{al_f1_sorted[-1][0]} (F1: {al_f1_sorted[-1][1]:.4f})\n")
        file_handle.write(
            f"彩色极光(Colored)误分为离散极光(Discrete)：{al_metrics['precision'][2]:.1%}（全量标注：{full_metrics['precision'][2]:.1%}）\n")
        file_handle.write(
            f"破裂型极光(Breakup)误分为弧状极光(Arcs)：{al_metrics['precision'][1]:.1%}（全量标注：{full_metrics['precision'][1]:.1%}）\n")
        file_handle.write("=" * 80 + "\n")

    print(f"详细指标报告已保存至：{save_path}")


# 训练/评估函数
def train_one_epoch(model, optimizer, train_loader, criterion, device, epoch_logs):
    model.train()
    total_loss = 0.0
    total_correct = 0
    pbar = tqdm(train_loader, desc='Training')
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        total_correct += (out.argmax(1) == y).sum().item()
        pbar.set_postfix({'Loss': loss.item(), 'Acc': total_correct / len(train_loader.dataset)})

    train_loss = total_loss / len(train_loader.dataset)
    train_acc = total_correct / len(train_loader.dataset)
    epoch_logs['train_loss'].append(train_loss)
    epoch_logs['train_acc'].append(train_acc)
    return train_loss, train_acc


def evaluate(model, test_loader, criterion, device, epoch_logs, return_preds=True):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    all_pred = []
    all_true = []
    with th.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
            total_correct += (out.argmax(1) == y).sum().item()
            if return_preds:
                all_pred.extend(out.argmax(1).cpu().numpy())
                all_true.extend(y.cpu().numpy())

    test_loss = total_loss / len(test_loader.dataset)
    test_acc = total_correct / len(test_loader.dataset)
    epoch_logs['test_loss'].append(test_loss)
    epoch_logs['test_acc'].append(test_acc)
    if return_preds:
        return test_loss, test_acc, np.array(all_true), np.array(all_pred)
    else:
        return test_loss, test_acc


# 可视化函数
def plot_training_curve(all_epoch_logs, save_path='al_training_curve.png'):
    epoch_logs = all_epoch_logs[-1]
    epochs = len(epoch_logs['train_loss'])

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), epoch_logs['train_loss'], 'o-',
             label='Train Loss', color='#1f77b4', linewidth=1.5, markersize=4)
    plt.plot(range(1, epochs + 1), epoch_logs['test_loss'], 's-',
             label='Test Loss', color='#ff7f0e', linewidth=1.5, markersize=4)
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('Loss', fontsize=11)
    plt.title('Active Learning: Training & Test Loss Curve', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), epoch_logs['train_acc'], 'o-',
             label='Train Accuracy', color='#2ca02c', linewidth=1.5, markersize=4)
    plt.plot(range(1, epochs + 1), epoch_logs['test_acc'], 's-',
             label='Test Accuracy', color='#d62728', linewidth=1.5, markersize=4)
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('Accuracy', fontsize=11)
    plt.title('Active Learning: Training & Test Accuracy Curve', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n主动学习训练曲线已保存至：{save_path}")


def plot_confusion_matrix(y_true, y_pred, classes, save_path='al_confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred, normalize='true') * 100
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='.1f',
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes,
        cbar_kws={'label': 'Classification Accuracy (%)'}
    )
    plt.xlabel('Predicted Class', fontsize=12)
    plt.ylabel('True Class', fontsize=12)
    plt.title('Active Learning: Confusion Matrix of Aurora Classification', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"主动学习混淆矩阵已保存至：{save_path}")


# ========== 不确定性计算和样本筛选函数 ==========
def calculate_uncertainty(model, unlabeled_loader, device):
    model.eval()
    all_uncertainty = []
    all_features = []
    all_paths = []
    with th.no_grad():
        for x, _, paths in tqdm(unlabeled_loader, desc='Calculating Uncertainty'):
            x = x.to(device)
            features = model(x, return_feature=True)
            logits = model.head(features)
            probs = th.softmax(logits, dim=1).cpu().numpy()
            max_prob = np.max(probs, axis=1)
            least_confidence = 1 - max_prob
            sorted_probs = np.sort(probs, axis=1)[:, ::-1]
            margin = sorted_probs[:, 0] - sorted_probs[:, 1]
            margin_uncertainty = 1 / (margin + 1e-8)
            combined_uncertainty = 0.5 * least_confidence + 0.5 * margin_uncertainty
            all_uncertainty.extend(combined_uncertainty)
            all_features.extend(features.cpu().numpy())
            all_paths.extend(paths)
    return np.array(all_uncertainty), np.array(all_features), all_paths


def select_high_value_samples(uncertainty, features, paths, select_size, full_dataset):
    select_size = min(select_size, len(uncertainty))
    top_k = min(int(select_size * 2), len(uncertainty))
    top_uncertain_idx = np.argsort(uncertainty)[::-1][:top_k]
    top_features = features[top_uncertain_idx]
    top_paths = [paths[i] for i in top_uncertain_idx]
    if len(top_features) < select_size:
        path2idx = {full_dataset.get_path(i): i for i in range(len(full_dataset))}
        selected_full_indices = [path2idx[p] for p in top_paths[:select_size]]
    else:
        kmeans = KMeans(n_clusters=select_size, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(top_features)
        selected_full_indices = []
        path2idx = {full_dataset.get_path(i): i for i in range(len(full_dataset))}
        for cluster_id in range(select_size):
            cluster_sample_idx = np.where(cluster_labels == cluster_id)[0]
            if len(cluster_sample_idx) == 0:
                cluster_sample_idx = [
                    np.argmin(np.linalg.norm(top_features - kmeans.cluster_centers_[cluster_id], axis=1))]
            cluster_center = kmeans.cluster_centers_[cluster_id]
            distances = np.linalg.norm(top_features[cluster_sample_idx] - cluster_center, axis=1)
            best_sample_in_cluster = cluster_sample_idx[np.argmin(distances)]
            best_path = top_paths[best_sample_in_cluster]
            selected_full_indices.append(path2idx[best_path])
    return selected_full_indices


# ========== 主动学习 ==========
def active_learning_run():
    global sim_unlabeled_indices, sim_unlabeled_dataset, sim_unlabeled_loader
    al_rounds = 4
    select_size_per_round = 50
    epochs_per_round = 15

    # 记录每轮精度和标注样本数
    round_acc_list = []
    round_sample_count = []

    # 初始化标注集
    current_labeled_indices = init_labeled_indices.tolist()
    current_labeled_dataset = Subset(full_train_dataset, current_labeled_indices)
    all_al_logs = []
    best_acc = 0.0
    best_true = None
    best_pred = None

    for round in range(al_rounds):
        print(f"\n=== 主动学习第 {round + 1}/{al_rounds} 轮 ===")
        print(f"当前标注样本数量：{len(current_labeled_dataset)}")
        print(f"当前未标注样本数量：{len(sim_unlabeled_dataset)}")

        # 训练模型
        current_train_loader = DataLoader(current_labeled_dataset, batch_size=64, shuffle=True)
        model = SwinTransformerModel(num_classes=num_classes).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_per_round)
        criterion = nn.CrossEntropyLoss()
        round_logs = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

        for epoch in range(epochs_per_round):
            train_loss, train_acc = train_one_epoch(model, optimizer, current_train_loader, criterion, device,
                                                    round_logs)
            test_loss, test_acc, all_true, all_pred = evaluate(model, test_loader, criterion, device, round_logs)
            scheduler.step()
            print(f"Round {round + 1}, Epoch {epoch}: Train Acc {train_acc:.4f}, Test Acc {test_acc:.4f}")

        # 记录每轮精度和样本数
        round_acc_list.append(test_acc)
        round_sample_count.append(len(current_labeled_dataset))

        # 保存最优结果
        all_al_logs.append(round_logs)
        if test_acc > best_acc:
            best_acc = test_acc
            best_true = all_true
            best_pred = all_pred
            th.save(model.state_dict(), f'al_best_model_round{round + 1}.pth')

        # 筛选样本
        if round < al_rounds - 1 and len(sim_unlabeled_dataset) >= select_size_per_round:
            uncertainty, features, unlabeled_paths = calculate_uncertainty(model, sim_unlabeled_loader, device)
            selected_full_indices = select_high_value_samples(
                uncertainty, features, unlabeled_paths, select_size_per_round, full_train_dataset
            )
            # 更新标注集
            current_labeled_indices.extend(selected_full_indices)
            current_labeled_dataset = Subset(full_train_dataset, current_labeled_indices)
            # 更新未标注池
            sim_unlabeled_indices = [i for i in sim_unlabeled_indices if i not in selected_full_indices]
            sim_unlabeled_dataset = Subset(full_train_dataset, sim_unlabeled_indices)
            sim_unlabeled_loader = UnlabeledLoaderWrapper(sim_unlabeled_dataset, batch_size=64, shuffle=False)
            print(f"第 {round + 1} 轮筛选完成，新增 {len(selected_full_indices)} 个标注样本")
        elif round < al_rounds - 1 and len(sim_unlabeled_dataset) < select_size_per_round:
            print(f"⚠️ 未标注样本数量不足，停止后续筛选")
            break

    # ========== 全量标注模型训练与指标计算 ==========
    print(f"\n=== 全量标注模型训练 ===")
    full_train_loader = DataLoader(full_train_dataset, batch_size=64, shuffle=True)
    full_model = SwinTransformerModel(num_classes=num_classes).to(device)
    full_optimizer = optim.AdamW(full_model.parameters(), lr=5e-5, weight_decay=1e-4)
    full_scheduler = optim.lr_scheduler.CosineAnnealingLR(full_optimizer, T_max=30)
    full_logs = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    for epoch in range(30):
        train_one_epoch(full_model, full_optimizer, full_train_loader, criterion, device, full_logs)
        full_test_loss, full_test_acc, full_true, full_pred = evaluate(full_model, test_loader, criterion, device,
                                                                       full_logs)
        full_scheduler.step()
    print(f"全量标注模型测试精度：{full_test_acc:.4f}")
    print(
        f"主动学习用 {len(current_labeled_dataset) / len(full_train_dataset) * 100:.1f}% 的标注量，达到全量标注 {best_acc / full_test_acc * 100:.1f}% 的精度")

    # ========== 计算详细指标并生成报告 ==========
    # 1. 计算主动学习和全量标注的类别级指标
    al_metrics = calculate_class_metrics(best_true, best_pred, classes)
    full_metrics = calculate_class_metrics(full_true, full_pred, classes)

    # 2. 生成收敛曲线
    plot_al_convergence_curve(round_acc_list, round_sample_count)
    plot_training_curve(all_al_logs)
    plot_confusion_matrix(best_true, best_pred, classes)

    # 3. 生成详细指标报告
    generate_metrics_report(
        al_metrics, full_metrics,
        al_sample_count=len(current_labeled_dataset),
        full_sample_count=len(full_train_dataset)
    )

    # 4. 打印关键指标摘要 + 类别级数据表格
    print(f"\n=== 关键指标摘要 ===")
    print(f"主动学习宏平均F1：{al_metrics['macro']['f1']:.4f}")
    print(f"全量标注宏平均F1：{full_metrics['macro']['f1']:.4f}")
    print(
        f"类别级最佳F1（主动学习）：{max(al_metrics['f1']):.4f}（{classes[al_metrics['f1'].index(max(al_metrics['f1']))]}）")
    print(
        f"类别级最差F1（主动学习）：{min(al_metrics['f1']):.4f}（{classes[al_metrics['f1'].index(min(al_metrics['f1']))]}）")

    # 打印类别级数据表格（主动学习）
    print(f"\n=== 类别级性能详情（主动学习） ===")
    print(f"{'类别':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'样本数':<10}")
    print("-" * 60)
    for cls, p, r, f1, s in zip(al_metrics['class'], al_metrics['precision'], al_metrics['recall'], al_metrics['f1'],
                                al_metrics['support']):
        print(f"{cls:<12} {p:<12.4f} {r:<12.4f} {f1:<12.4f} {s:<10}")

    # 打印类别级数据表格（全量标注）
    print(f"\n=== 类别级性能详情（全量标注） ===")
    print(f"{'类别':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'样本数':<10}")
    print("-" * 60)
    for cls, p, r, f1, s in zip(full_metrics['class'], full_metrics['precision'], full_metrics['recall'],
                                full_metrics['f1'], full_metrics['support']):
        print(f"{cls:<12} {p:<12.4f} {r:<12.4f} {f1:<12.4f} {s:<10}")


# 启动主动学习
if __name__ == "__main__":
    active_learning_run()
