def compute_seg_metrics(pred, target, class_idx):
    pred = pred == class_idx
    target = target == class_idx
    TP = (pred & target).sum()
    FP = (pred & ~target).sum()
    FN = (~pred & target).sum()

    iou = TP / (TP + FP + FN) if TP + FP + FN > 0 else 0
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    return precision, recall, iou


def accuracy(predictions, targets):
    correct = (predictions == targets).sum().item()
    total = targets.numel()
    return correct / total


def topk_accuracy(breed_preds_indices, true_breed, k=3):
    correct_count = 0
    for i, tb in enumerate(true_breed):
        if tb.item() in breed_preds_indices[i]:
            correct_count += 1
    return correct_count / len(true_breed)


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.cpu().numpy()
    plt.figure(figsize=(10, 10))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def mask_to_img(mask):
    mask_img = torch.zeros((mask.shape[0], 128, 128, 3)).to(device)
    mask_img[mask == 0] = torch.tensor([1., 0., 0.], device=device)
    mask_img[mask == 1] = torch.tensor([0., 1., 0.], device=device)
    mask_img = mask_img.permute(0, 3, 1, 2)
    return mask_img


def dice_loss(pred, target, smooth=1.0):
    n, c, h, w = pred.size()
    target_one_hot = F.one_hot(target, num_classes=c).permute(0, 3, 1, 2)

    pred = F.softmax(pred, dim=1)
    intersection = (pred * target_one_hot).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()



