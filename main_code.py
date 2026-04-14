import os
import sys
import random
import torch as th
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import timm
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix, roc_curve, auc, RocCurveDisplay
import seaborn as sns
from sklearn.model_selection import KFold
import pandas as pd  



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


train_dir = "/home/zhuby/极光图像分类/TrainingImages"
test_dir = "/home/zhuby/极光图像分类/TestImages"
results_dir = "/home/zhuby/极光图像分类/Results_CV"

os.makedirs(results_dir, exist_ok=True)



class AuroraDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_paths = sorted([
            os.path.join(img_dir, f) for f in os.listdir(img_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        self.labels = []
        for path in self.img_paths:
            filename = os.path.basename(path)
            try:
                class_str = filename.split('class_')[-1].split('.')[0]
                self.labels.append(int(class_str))
            except Exception:
                raise ValueError(f"无法从文件名 {filename} 解析标签")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label, img_path



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

if not os.path.exists(train_dir) or not os.path.exists(test_dir):
    print(f"❌ 错误：找不到数据目录。")
    sys.exit(1)

full_train_dataset = AuroraDataset(train_dir, transform=train_transform)
test_dataset = AuroraDataset(test_dir, transform=test_transform)

print(f"✅ 数据加载完成：训练集 {len(full_train_dataset)} 张，测试集 {len(test_dataset)} 张")

device = 'cuda' if th.cuda.is_available() else 'cpu'
classes = ['Arcs', 'Breakup', 'Colored', 'Discrete', 'Edge', 'Faint', 'Patchy']
num_classes = len(classes)



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

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x



class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        def forward_hook(module, input, output): self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output): self.gradients = grad_output[0].detach()

        self.f_hook = self.target_layer.register_forward_hook(forward_hook)
        self.b_hook = self.target_layer.register_backward_hook(backward_hook)

    def remove_hooks(self):
        self.f_hook.remove();
        self.b_hook.remove()

    def __call__(self, input_tensor, target_class=None):
        self.model.eval()
        input_tensor = input_tensor.unsqueeze(0).to(device)
        input_tensor.requires_grad_(True)
        output = self.model(input_tensor)
        if target_class is None: target_class = output.argmax().item()
        self.model.zero_grad()
        one_hot = th.zeros_like(output)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot, retain_graph=False)
        if self.gradients is None or self.activations is None: return np.zeros((32, 32))
        weights = th.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = th.sum(weights * self.activations, dim=1)[0]
        cam = th.relu(cam).cpu().numpy()
        self.remove_hooks()
        return cam


def generate_attention_heatmap(model, image_tensor, original_image_path, class_names, save_dir):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    try:
        original_img = Image.open(original_image_path).convert('RGB')
        original_img_np = np.array(original_img)
        h_orig, w_orig = original_img_np.shape[:2]
    except:
        return None
    with th.no_grad():
        input_tensor = image_tensor.unsqueeze(0).to(device)
        output = model(input_tensor)
        pred_class = output.argmax(1).item()
        pred_prob = th.softmax(output, dim=1)[0, pred_class].item()
    filename = os.path.basename(original_image_path)
    try:
        true_class = int(filename.split('class_')[-1].split('.')[0])
    except:
        true_class = -1
    target_layer = model.backbone.patch_embed.proj
    cam = np.zeros((32, 32))
    try:
        grad_cam = GradCAM(model, target_layer)
        cam = grad_cam(image_tensor, target_class=pred_class)
    except:
        pass
    cam = np.maximum(cam, 0)
    if cam.max() > 0: cam = cam / cam.max()
    heatmap = cv2.resize(cam, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)
    heatmap = cv2.GaussianBlur(heatmap.astype(np.float32), (11, 11), 0)
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    superimposed_img = np.clip(heatmap_color * 0.5 + original_img_np * 0.5, 0, 255).astype(np.uint8)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(original_img_np)
    label_color = 'green' if true_class == pred_class else 'red'
    true_name = class_names[true_class] if 0 <= true_class < len(classes) else "Unknown"
    axes[0].set_title(f'Original\nTrue: {true_name}', fontsize=12, fontweight='bold', color=label_color)
    axes[0].axis('off')
    im = axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap', fontsize=22, fontweight='bold')
    axes[1].axis('off')
    cbar = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=20)
    axes[2].imshow(superimposed_img)
    pred_name = class_names[pred_class] if 0 <= pred_class < len(classes) else "Unknown"
    axes[2].set_title(f'Overlay\nPred: {pred_name} ({pred_prob:.3f})', fontsize=14, fontweight='bold',
                      color=label_color)
    axes[2].axis('off')
    plt.tight_layout()
    filename_base = os.path.basename(original_image_path).split('.')[0]
    save_path = os.path.join(save_dir, f'{filename_base}_gradcam.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    return save_path


def generate_heatmaps_for_all_test_images(model, dataloader, class_names, save_dir='heatmaps'):
    os.makedirs(save_dir, exist_ok=True)
    generated_count = 0
    total_images = len(dataloader.dataset)
    print(f"正在为所有 {total_images} 张测试图像生成 Grad-CAM 热图...")
    model.eval()
    for images, labels, img_paths in tqdm(dataloader, desc='Generating heatmaps'):
        for i in range(images.size(0)):
            try:
                generate_attention_heatmap(model, images[i], img_paths[i], class_names, save_dir)
                generated_count += 1
            except:
                continue
    print(f"\n✅ 热图生成完成！成功生成 {generated_count} 张热图")



def train_one_epoch(model, optimizer, train_loader, criterion, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    pbar = tqdm(train_loader, desc='Training', leave=False)
    for x, y, _ in pbar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        total_correct += (out.argmax(1) == y).sum().item()
        pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
    return total_loss / len(train_loader.dataset), total_correct / len(train_loader.dataset)


def evaluate_with_probs(model, loader, criterion, device):
    """返回 loss, acc, true_labels, pred_labels, 以及 probabilities (用于 ROC)"""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    all_pred = []
    all_true = []
    all_probs = []

    with th.no_grad():
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            probs = th.softmax(out, dim=1)

            total_loss += loss.item() * x.size(0)
            total_correct += (out.argmax(1) == y).sum().item()
            all_pred.extend(out.argmax(1).cpu().numpy())
            all_true.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return total_loss / len(loader.dataset), total_correct / len(loader.dataset), \
        np.array(all_true), np.array(all_pred), np.array(all_probs)


def plot_confusion_matrix(y_true, y_pred, classes, save_path='confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred, normalize='true') * 100
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.1f', cmap='Blues', xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Classification Accuracy (%)'})
    plt.xlabel('Predicted Class', fontsize=12)
    plt.ylabel('True Class', fontsize=12)
    plt.title('Confusion Matrix of Aurora Classification', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ 混淆矩阵已保存至：{save_path}")


def plot_cv_training_curve(fold_logs, save_path='cv_training_curve.png'):
    if not fold_logs: return
    max_epochs = len(fold_logs[0]['train_loss'])
    avg_train_loss = np.mean([log['train_loss'] for log in fold_logs], axis=0)
    avg_val_loss = np.mean([log['val_loss'] for log in fold_logs], axis=0)
    avg_train_acc = np.mean([log['train_acc'] for log in fold_logs], axis=0)
    avg_val_acc = np.mean([log['val_acc'] for log in fold_logs], axis=0)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, max_epochs + 1), avg_train_loss, 'o-', label='Avg Train Loss', color='#1f77b4')
    plt.plot(range(1, max_epochs + 1), avg_val_loss, 's-', label='Avg Val Loss', color='#ff7f0e')
    plt.xlabel('Epoch');
    plt.ylabel('Loss');
    plt.title('Average Training & Validation Loss (5-Fold)')
    plt.legend();
    plt.grid(alpha=0.3)
    plt.subplot(1, 2, 2)
    plt.plot(range(1, max_epochs + 1), avg_train_acc, 'o-', label='Avg Train Acc', color='#2ca02c')
    plt.plot(range(1, max_epochs + 1), avg_val_acc, 's-', label='Avg Val Acc', color='#d62728')
    plt.xlabel('Epoch');
    plt.ylabel('Accuracy');
    plt.title('Average Training & Validation Accuracy (5-Fold)')
    plt.legend();
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ K-Fold 训练曲线已保存至：{save_path}")



def plot_multiclass_roc(y_true, y_scores, classes, save_path='roc_curves.png'):
    plt.figure(figsize=(10, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))


    all_fpr = np.unique(np.concatenate([roc_curve(y_true == i, y_scores[:, i])[0] for i in range(len(classes))]))
    mean_tpr = np.zeros_like(all_fpr)
    aucs = []

    for i, color in zip(range(len(classes)), colors):
        fpr, tpr, _ = roc_curve(y_true == i, y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        mean_tpr += np.interp(all_fpr, fpr, tpr)

        plt.plot(fpr, tpr, color=color, lw=2, label=f'{classes[i]} (AUC={roc_auc:.4f})')

    mean_tpr /= len(classes)
    mean_auc = auc(all_fpr, mean_tpr)


    plt.plot([0, 1], [0, 1], 'k--', lw=2, label=f'Random (AUC=0.5)')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=20)
    plt.ylabel('True Positive Rate', fontsize=20)
    plt.title('Class-wise ROC Curves', fontsize=20)
    plt.xticks(fontsize=18)  # 设置 X 轴刻度数字大小
    plt.yticks(fontsize=18)  # 设置 Y 轴刻度数字大小
    plt.legend(loc="lower right", fontsize=18)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ ROC 曲线已保存至：{save_path}")
    return aucs


# 5-Fold Cross-Validation
K_FOLDS = 5
EPOCHS = 30
BATCH_SIZE = 64
criterion = nn.CrossEntropyLoss()

fold_test_scores = []
fold_logs = []
best_models_states = []
fold_test_probs = []  

kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

print(f"\n🚀 开始 {K_FOLDS}-Fold 交叉验证 (Device: {device})")

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

for fold, (train_idx, val_idx) in enumerate(kf.split(full_train_dataset)):
    print(f"\n{'=' * 40}")
    print(f"📂 正在进行 Fold {fold + 1}/{K_FOLDS}")
    print(f"{'=' * 40}")

    train_subset = Subset(full_train_dataset, train_idx)
    val_subset = Subset(full_train_dataset, val_idx)
    train_loader_fold = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader_fold = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = SwinTransformerModel(num_classes=num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    current_fold_logs = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, optimizer, train_loader_fold, criterion, device)
        val_loss, val_acc, _, _, _ = evaluate_with_probs(model, val_loader_fold, criterion, device)
        scheduler.step()

        current_fold_logs['train_loss'].append(train_loss)
        current_fold_logs['train_acc'].append(train_acc)
        current_fold_logs['val_loss'].append(val_loss)
        current_fold_logs['val_acc'].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1}: Val Acc={val_acc:.4f}")

    fold_logs.append(current_fold_logs)
    if best_model_state is None: continue

    model.load_state_dict(best_model_state)
    best_models_states.append(best_model_state)


    test_loss, test_acc, test_true, test_pred, test_probs = evaluate_with_probs(model, test_loader, criterion, device)
    precision, recall, f1, _ = score(test_true, test_pred, average=None, labels=range(num_classes))  # 按类别计算

    fold_test_scores.append({
        'acc': test_acc,
        'precision_per_class': precision,
        'recall_per_class': recall,
        'f1_per_class': f1,
        'true': test_true,
        'pred': test_pred,
        'probs': test_probs
    })
    print(f"  📊 Fold {fold + 1} Test Acc: {test_acc:.4f}")


if not fold_test_scores:
    print("\n❌ 致命错误：没有任何 Fold 成功完成。")
    sys.exit(1)

print("\n" + "=" * 60)
print("🏆 5-Fold 交叉验证最终结果汇总")
print("=" * 60)


avg_precision = np.mean([s['precision_per_class'] for s in fold_test_scores], axis=0)
avg_recall = np.mean([s['recall_per_class'] for s in fold_test_scores], axis=0)
avg_f1 = np.mean([s['f1_per_class'] for s in fold_test_scores], axis=0)
avg_acc = np.mean([s['acc'] for s in fold_test_scores])

# Category | Precision | Recall | F1-Score
data = {
    'Category': classes,
    'Precision': avg_precision,
    'Recall': avg_recall,
    'F1-Score': avg_f1
}
df = pd.DataFrame(data)

pd.set_option('display.precision', 4)
pd.set_option('display.colheader_justify', 'center')
print("\n📊 分类性能详细报告 (Average over 5 Folds):")
print(df.to_string(index=False))
print(f"\n🌟 整体平均准确率 (Mean Accuracy): {avg_acc:.4f}")


csv_path = os.path.join(results_dir, 'classification_report.csv')
df.to_csv(csv_path, index=False)
print(f"✅ 详细报告已保存至：{csv_path}")


best_fold_idx = np.argmax([s['acc'] for s in fold_test_scores])
best_result = fold_test_scores[best_fold_idx]
best_model_state = best_models_states[best_fold_idx]

print(f"\n📈 选用 Fold {best_fold_idx + 1} 的模型绘制详细图表 (Test Acc: {best_result['acc']:.4f})")

# 混淆矩阵
cm_path = os.path.join(results_dir, 'confusion_matrix_cv.png')
plot_confusion_matrix(best_result['true'], best_result['pred'], classes, save_path=cm_path)

# ROC 曲线
roc_path = os.path.join(results_dir, 'roc_curves_cv.png')
plot_multiclass_roc(best_result['true'], best_result['probs'], classes, save_path=roc_path)


curve_path = os.path.join(results_dir, 'cv_training_curve.png')
plot_cv_training_curve(fold_logs, save_path=curve_path)


final_model = SwinTransformerModel(num_classes=num_classes).to(device)
final_model.load_state_dict(best_model_state)
heatmaps_dir = os.path.join(results_dir, 'heatmaps_cv')
generate_heatmaps_for_all_test_images(final_model, test_loader, classes, save_dir=heatmaps_dir)

print(f"\n🎉 所有任务成功完成！结果保存在：{results_dir}")
    active_learning_run()
