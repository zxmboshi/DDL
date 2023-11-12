import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataloaders.la_heart import LAHeart
from dataloaders.util import TwoStreamBatchSampler, RandomCrop, CenterCrop, RandomRotFlip, ToTensor
from torchvision import transforms as T
from networks.vnet import VNet


def cal_dice(output, target, eps=1e-3):
    output = torch.argmax(output, dim=1)
    inter = torch.sum(output * target) + eps
    union = torch.sum(output) + torch.sum(target) + eps * 2
    dice = 2 * inter / union
    return dice


def train_loop(model, optimizer, criterion, train_loader, device):
    model.train()
    running_loss = dice_train = 0
    pbar = tqdm(train_loader)

    for sample in pbar:
        image, label = sample['image'][:2], sample['label'][:2]
        image, label = image.to(device), label.to(device)

        outputs = model(image)
        loss = criterion(outputs, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        dice = cal_dice(outputs, label)
        pbar.set_postfix(loss="{:.3f}".format(loss.item()), dice="{:.3f}".format(dice.item()))

        running_loss += loss.item()
        dice_train += dice.item()

    loss = running_loss / len(train_loader)
    dice = dice_train / len(train_loader)
    return {'loss': loss, 'dice': dice}


def eval_loop(model, criterion, valid_loader, device):
    model.eval()
    running_loss = dice_valid = 0
    pbar = tqdm(valid_loader)

    for sample in pbar:
        image, label = sample['image'], sample['label']
        image, label = image.to(device), label.to(device)

        with torch.no_grad():
            outputs = model(image)
            loss = criterion(outputs, label)

        dice = cal_dice(outputs, label)
        pbar.set_postfix(loss="{:.3f}".format(loss.item()), dice="{:.3f}".format(dice.item()))

        running_loss += loss.item()
        dice_valid += dice.item()

    loss = running_loss / len(valid_loader)
    dice = dice_valid / len(valid_loader)
    return {'loss': loss, 'dice': dice}


def train(args, model, optimizer, scheduler, criterion, train_loader, valid_loader, epochs, device):
    loss_min = 999.0
    for e in range(epochs):
        # train for epoch
        train_metrics = train_loop(model, optimizer, criterion, train_loader, device)
        
        if e % 10 == 9:
            # eval for epoch
            valid_metrics = eval_loop(model, criterion, valid_loader, device)
            if valid_metrics['loss'] < loss_min:
                loss_min = valid_metrics['loss']
                # save model
                torch.save(model.state_dict(), os.path.join(args.save_path, 'best.pth'))

            info = "Epoch:[{:0>3d}/{}] lr: {:.4f} train_loss: {:.3f} train_dice: {:.3f} valid_loss: {:.3f} valid_dice {:.3f}".format(
                e + 1, epochs,
                optimizer.param_groups[0][
                    'lr'],
                train_metrics['loss'],
                train_metrics['dice'],
                valid_metrics['loss'],
                valid_metrics['dice'])
        else:
            info = "Epoch:[{:0>3d}/{}] lr: {:.4f} train_loss: {:.3f} train_dice: {:.3f}".format(
                e + 1, epochs,
                optimizer.param_groups[0][
                    'lr'],
                train_metrics['loss'],
                train_metrics['dice'])

        print(info)
        with open(os.path.join(args.save_path, 'log.txt'), 'a') as f:
            f.write(info + '\n')

        scheduler.step()

    print("Finished Training!")


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.deterministic:
        torch.manual_seed(args.seed)  # 为CPU设置种子用于生成随机数，以使得结果是确定的
        torch.cuda.manual_seed_all(args.seed)  # 为所有的GPU设置种子，以使得结果是确定的
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

    train_trans = T.Compose([
        RandomCrop(args.patch_size),
        RandomRotFlip(),
        ToTensor()
    ])
    valid_trans = T.Compose([
        CenterCrop(args.patch_size),
        ToTensor()
    ])

    labeled_idxs = list(range(args.labelnum))
    unlabeled_idxs = list(range(args.labelnum, 123))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, 4, 2)

    # data info
    db_train = LAHeart(base_dir=args.data_path, split='train', transform=train_trans)
    db_valid = LAHeart(base_dir=args.data_path, split='test', transform=valid_trans)
    print('Using {} images for training, {} images for validation.'.format(len(db_train), len(db_valid)))

    # trainloader = DataLoader(db_train, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True)
    validloader = DataLoader(db_valid, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = VNet(n_channels=args.in_ch, n_classes=args.num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr / 10)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    train(args, model, optimizer, scheduler, criterion, trainloader, validloader, args.epochs, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='Ink Detection', help='dataset_name')
    parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument('--labelnum', type=int,  default=25, help='labeled samples')
    parser.add_argument('--data_path', type=str, default='/data/Yb/data/LASet')
    parser.add_argument('--in_ch', type=int, default=1)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--patch_size', type=float, default=(112, 112, 80))
    parser.add_argument('--save_path', type=str, default='results/label24_sup')

    args = parser.parse_args()

    main(args)
