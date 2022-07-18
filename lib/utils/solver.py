import torch, glob, os, numpy as np
import sys

def is_power2(num):
    return num != 0 and ((num & (num - 1)) == 0)


def is_multiple(num, multiple):
    return num != 0 and num % multiple == 0


def step_learning_rate(optimizer, base_lr, epoch, step_epoch, multiplier=0.1, clip=1e-6):
    """Sets the learning rate to the base LR decayed by 10 every step epochs"""
    lr = max(base_lr * (multiplier ** (epoch // step_epoch)), clip)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def checkpoint_save(model, exp_path, exp_name, epoch, save_freq=16, use_cuda=True, logger=None):
    f = os.path.join(exp_path, exp_name + '-%09d'%epoch + '.pth')
    logger.info('Saving ' + f)
    model.cpu()
    torch.save(model.state_dict(), f)
    if use_cuda:
        model.cuda()

    #remove previous checkpoints unless they are a power of 2 or a multiple of 16 to save disk space
    epoch = epoch - 1
    f = os.path.join(exp_path, exp_name + '-%09d'%epoch + '.pth')
    if os.path.isfile(f):
        if not is_multiple(epoch, save_freq) and not is_power2(epoch):
            os.remove(f)


def checkpoint_restore(model, exp_path, exp_name, use_cuda=True, epoch=0, dist=False, f='', logger=None):
    if use_cuda:
        model.cpu()
    if not f:
        if epoch > 0:
            f = os.path.join(exp_path, exp_name + '-%09d'%epoch + '.pth')
            assert os.path.isfile(f)
        else:
            f = sorted(glob.glob(os.path.join(exp_path, exp_name + '-*.pth')))
            if len(f) > 0:
                f = f[-1]
                epoch = int(f[len(exp_path) + len(exp_name) + 2 : -4])

    if len(f) > 0:
        logger.info('Restore from ' + f)
        checkpoint = torch.load(f)
        for k, v in checkpoint.items():
            if 'module.' in k:
                checkpoint = {k[len('module.'):]: v for k, v in checkpoint.items()}
            break
        if dist:
            model.module.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)

    if use_cuda:
        model.cuda()
    return epoch + 1