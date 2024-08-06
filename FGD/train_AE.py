import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from FGD.embedding_net import EmbeddingNet
from FGD.embedding_space_evaluator import EmbeddingSpaceEvaluator

from loader import DatasetBVHLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def train_iter(target_data, net, optim):
    # zero gradients
    optim.zero_grad()

    # reconstruction loss
    feat, recon_data = net(target_data)
    recon_loss = F.l1_loss(recon_data, target_data, reduction='none')
    recon_loss = torch.mean(recon_loss, dim=(1, 2))

    if True:  # use pose diff
        target_diff = target_data[:, 1:] - target_data[:, :-1]
        recon_diff = recon_data[:, 1:] - recon_data[:, :-1]
        recon_loss += torch.mean(F.l1_loss(recon_diff, target_diff, reduction='none'), dim=(1, 2))

    recon_loss = torch.sum(recon_loss)

    recon_loss.backward()
    optim.step()

    ret_dict = {'loss': recon_loss.item()}
    return ret_dict

def evaluate_model(model, data_loader):
    n_loss, sum_loss = 0, 0
    with torch.no_grad():
        for iter_idx, target in enumerate(data_loader, 0):
            target_vec = target[0].float().to(device)
            feat, recon_data = model(target_vec)
            
            n_loss += target_vec.size(dim=0)
            sum_loss += compute_loss(target_vec, recon_data) * target_vec.size(dim=0)
    return sum_loss/n_loss

def compute_loss(target, recon):
    recon_loss = F.l1_loss(recon, target, reduction='none')
    recon_loss = torch.mean(recon_loss, dim=(1, 2))
    if True:  # use pose diff
        target_diff = target[:, 1:] - target[:, :-1]
        recon_diff = recon[:, 1:] - recon[:, :-1]
        recon_loss += torch.mean(F.l1_loss(recon_diff, target_diff, reduction='none'), dim=(1, 2))
    recon_loss = torch.sum(recon_loss)
    return recon_loss.item()

def main(trn_loader, 
         val_loader,
         generator,
         gen_optimizer,
         loss_meters,
         batch_size,
         print_interval,
         pose_dim,
         device,):
    min_val_loss = np.inf
    evaluator = EmbeddingSpaceEvaluator(None, None, None, dummy=True) # Dummy evaluator just to compute FGD
    with open("./FGD/output/training_log.txt", "a") as log_file:
        for epoch in range(1000):
            for iter_idx, target in enumerate(trn_loader, 0):
                target_vec = target[0].float().to(device)
                loss = train_iter(target_vec, generator, gen_optimizer)
                # loss values
                for loss_meter in loss_meters:
                    name = loss_meter.name
                    if name in loss:
                        loss_meter.update(loss[name], batch_size)

                # print training status
                if (iter_idx + 1) % print_interval == 0:
                    print_summary = 'EP {} ({:3d}) | '.format(epoch, iter_idx + 1)
                    for loss_meter in loss_meters:
                        if loss_meter.count > 0:
                            print_summary += '{}: {:.3f}, '.format(loss_meter.name, loss_meter.avg)
                            last_loss = loss_meter.avg
                            loss_meter.reset()
                    print(print_summary)
                    
            print(f'Epoch {epoch} validation')
            val_loss = evaluate_model(generator, val_loader)
            gt_feat, gt_labels = evaluator.run_samples(generator, trn_loader, device)
            tst_feat, tst_labels = evaluator.run_samples(generator, val_loader, device)
            frechet_dist = evaluator.frechet_distance(gt_feat, tst_feat)
            print(f'Whole val data avg loss: {val_loss}. FGD: {frechet_dist}')
            log_file.write(f'Epoch: {epoch}. Trn loss: {last_loss}. Val loss: {val_loss}. FGD: {frechet_dist}\n')
            log_file.flush()
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                # save model
                gen_state_dict = generator.state_dict()
                save_name = f'./FGD/output/genea_model_checkpoint_{trn_loader.dataset.window}_{pose_dim}.bin'
                torch.save({'pose_dim': pose_dim, 'n_frames': trn_loader.dataset.window, 'gen_dict': gen_state_dict}, save_name)


if __name__ == '__main__':
    # Hard-code for the TWH (Genea) dataset
    batch_size = 64

    trn_dataset = DatasetBVHLoader(name='trn', 
                                    path='./dataset/Genea2023/trn/main-agent/bvh', 
                                    load=True,
                                    step=10, 
                                    window=120)
    trn_loader = DataLoader(dataset=trn_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = DatasetBVHLoader(name='val', 
                                   path='./dataset/Genea2023/val/main-agent/bvh', 
                                   load=True,
                                   step=10, 
                                   window=120)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

    pose_dim = trn_dataset.__getitem__(0)[0].shape[1]
    generator = EmbeddingNet(pose_dim, trn_dataset.window).to(device)
    gen_optimizer = optim.Adam(generator.parameters(), lr=0.001, betas=(0.5, 0.999))
    loss_meters = [AverageMeter('loss')]
    print_interval = int(len(trn_loader) / 5)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main(
        trn_loader,
        val_loader,
        generator,
        gen_optimizer,
        loss_meters,
        batch_size,
        print_interval,
        pose_dim,
        device,
    )