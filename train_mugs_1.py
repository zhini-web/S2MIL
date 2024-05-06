import sys
import os
import numpy as np
import argparse
import random
import openslide
import PIL.Image as Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as functional
import torchvision.models as models
import torch.optim.lr_scheduler as lr_scheduler
import time
from sklearn.metrics import balanced_accuracy_score, recall_score

from tqdm import tqdm as tqdm
# from torchsummary import summary
# from eca.eca_model import eca_resnet50
# from sa.sa_resnet import sa_resnet50
from collections import OrderedDict
# from fcanet.fcanet import fcanet50
import warnings
from mugs.vision_transformer import vit_small
import mugs.utils as utils

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='MIL-nature-medicine-2019 tile classifier training script')
parser.add_argument('--train_lib', type=str, default='../dataset/c16/cnn_train_lib_4090.db',
                    help='path to train MIL library binary')

parser.add_argument('--val_lib', type=str, default='../dataset/c16/cnn_val_lib_4090.db',
                    help='path to validation MIL library binary. If present.')
parser.add_argument('--output', type=str, default='../WSI_mugs/train1', help='name of output file')

# parser.add_argument('--net', type=str, default='swin', help='name of model')
parser.add_argument('--resume', type=str, default=False, help='resume checkpoint')
parser.add_argument('--batch_size', type=int, default=128, help='mini-batch size (default: 512)')
parser.add_argument('--nepochs', type=int, default=50, help='number of epochs')
parser.add_argument('--workers', default=8, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--test_every', default=2, type=int, help='test on val every (default: 10)')
parser.add_argument('--weights', default=0.6, type=float,
                    help='unbalanced positive class weight (default: 0.5, balanced classes)')
parser.add_argument('--k', default=2, type=int,
                    help='top k tiles are assumed to be of the same class as the slide (default: 1, standard MIL)')
best_acc = 0


def Parallel2Single(origin_state):
    converted = OrderedDict()

    for k, v in origin_state.items():
        name = k[7:]
        converted[name] = v

    return converted


def seed_everything(seed):
    '''
    设置整个开发环境的seed
    :param seed:
    :param device:
    :return:
    '''
    import os
    import random
    import numpy as np

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True



def main():
    global args, best_acc
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    seed = 2023
    seed_everything(seed)

    c = '../mugs/checkpoint0040.pth'
    ckpt_path = c
    model = vit_small(num_classes=2)
    utils.load_pretrained_weights(model, ckpt_path, 'teacher', 'vit_small', 16)

    if args.weights == 0.5:
        criterion = nn.CrossEntropyLoss().to(device)
    else:
        w = torch.Tensor([1 - args.weights, args.weights])
        criterion = nn.CrossEntropyLoss(w).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=[0, 1])
        model.to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_trans = transforms.Compose([
                                      transforms.Resize(256, interpolation=3),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      normalize])
    val_trans = transforms.Compose([
                                      transforms.Resize(256, interpolation=3),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      normalize])
    train_dset = MILdataset(args.train_lib, train_trans)
    train_loader = torch.utils.data.DataLoader(
        train_dset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)


    if args.val_lib:
        val_dset = MILdataset(args.val_lib, val_trans)
        val_loader = torch.utils.data.DataLoader(
            val_dset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=False)
    time_mark = time.strftime('%Y_%m_%d_', time.localtime(time.time()))
    # open output file
    fconv = open(os.path.join(args.output, time_mark + 'CNN_convergence_2.csv'), 'w')
    fconv.write(' ,Train,,,,Validation,,,\n')
    fconv.write('epoch,acc,recall,fnr,loss,acc,recall,fnr')
    fconv.close()

    topk_list = []
    early_stop_count = 0


    start_epoch = 1
    if args.resume:
        path_checkpoint = ''
        checkpoint = torch.load(path_checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']

    # loop through epochs
    for epoch in range(start_epoch, args.nepochs + 1):
        if epoch >= args.nepochs * 2 / 3 and early_stop_count > 3:
            print('Early stop at Epoch:' + str(epoch + 1))
            break
        start_time = time.time()
        # Train
        train_dset.setmode(1)
        train_probs = inference(epoch, train_loader, model, 'train', device)
        topk = group_argtopk(np.array(train_dset.slideIDX), train_probs, args.k)
        if epoch >= 10:

            topk_last = topk_list[-1]
            if sum(np.not_equal(topk_last, topk)) < 0.01 * len(topk):
                early_stop_count += 1
        topk_list.append(topk.copy())
        train_dset.maketraindata(topk)
        train_dset.shuffletraindata()
        train_dset.setmode(2)
        whole_acc, whole_recall, whole_fnr, whole_loss = train(epoch, train_loader, model, criterion, optimizer, device)

        print('Training\tEpoch: [{}/{}] Acc: {}  Recall:{}  Fnr:{}  Loss: {}'.format(epoch, \
                                                                                     args.nepochs, whole_acc,
                                                                                     whole_recall, whole_fnr,
                                                                                     whole_loss))
        result = '\n' + str(epoch) + ',' + str(whole_acc) + ',' + str(whole_recall) + ',' + str(
            whole_fnr) + ',' + str(whole_loss)
        if args.val_lib:
            val_dset.setmode(1)
            val_probs = inference(epoch, val_loader, model, 'val', device)
            maxs = group_max(np.array(val_dset.slideIDX), val_probs, len(val_dset.targets))
            pred = [1 if x >= 0.5 else 0 for x in maxs]
            metrics_meters = calc_accuracy(pred, val_dset.targets)
            str_logs = ['{} - {:.4}'.format(k, v) for k, v in metrics_meters.items()]
            s = ', '.join(str_logs)
            print('Validation\tEpoch: [{}/{}]  '.format(epoch, args.nepochs) + s)
            result = result + ',' + str(metrics_meters['acc']) + ',' + str(metrics_meters['recall']) + ',' \
                     + str(metrics_meters['fnr'])
            fconv = open(os.path.join(args.output, time_mark + 'CNN_convergence_2.csv'), 'a')
            fconv.write(result)
            fconv.close()
            # Save best model
            tmp_acc = (metrics_meters['acc'] + metrics_meters['recall']) / 2 - metrics_meters['fnr']
            print("tmp_acc:", tmp_acc)
            if (tmp_acc >= best_acc or metrics_meters['acc'] > 0.80 or epoch == args.nepochs) and epoch > 0:
                best_acc = tmp_acc.copy()
                print("val_acc: ", metrics_meters['acc'])

                obj = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict()
                }
                torch.save(obj, os.path.join("../WSI_mugs/train1",
                                             str(epoch) + '_CNN_checkpoint_best.pth'))
        print('\tEpoch {} has been finished, needed {} sec.'.format(epoch, time.time() - start_time))


def inference(run, loader, model, phase, device):
    model.eval()
    probs = torch.FloatTensor(len(loader.dataset))
    logs = {}
    whole_probably = 0.

    with torch.no_grad():
        with tqdm(loader, desc='Epoch ' + str(run) + ': ' + phase + '\'s inferencing', file=sys.stdout, disable=False) as iterator:
            for i, input in enumerate(iterator):
                input = input.to(device)
                output = F.softmax(model(input), dim=1)
                prob = output.detach()[:, 1].clone()
                probs[i * args.batch_size:i * args.batch_size + input.size(0)] = prob
                avg_prob = np.sum(prob.cpu().numpy()) / args.batch_size
                whole_probably = whole_probably + avg_prob
                temp_log = {'average tumor probably': avg_prob}
                logs.update(temp_log)
                str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
                s = ', '.join(str_logs)
                iterator.set_postfix_str(s)
            whole_probably = whole_probably / (i + 1)
            iterator.set_postfix_str('Whole average probably is ' + str(whole_probably))
    return probs.cpu().numpy()


def train(run, loader, model, criterion, optimizer, device):
    model.train()
    whole_loss = 0.
    whole_acc = 0.
    whole_recall = 0.
    whole_fnr = 0.
    logs = {}

    with tqdm(loader, desc='Epoch ' + str(run) + ' is trainng', file=sys.stdout, disable=False) as iterator:
        for i, (input, target) in enumerate(iterator):
            input, target = input.to(device), target.to(device)
            output = model(input)
            loss = criterion(output, target)
            _, pred = torch.max(output, 1)  # 计算准确率
            pred = pred.data.cpu().numpy()
            metrics_meters = calc_accuracy(pred, target.cpu().numpy())
            logs.update(metrics_meters)
            logs.update({'loss': loss.item()})
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
            s = ', '.join(str_logs)
            iterator.set_postfix_str(s)

            whole_acc += metrics_meters['acc']
            whole_recall += metrics_meters['recall']
            whole_fnr += metrics_meters['fnr']
            whole_loss += loss.item()
    return round(whole_acc / (i + 1), 3), round(whole_recall / (i + 1), 3), round(whole_fnr / (i + 1), 3), round(
        whole_loss / (i + 1), 3)


def calc_accuracy(pred, real):
    if str(type(pred)) != "<class 'numpy.ndarray'>":
        pred = np.array(pred)
    if str(type(real)) != "<class 'numpy.ndarray'>":
        real = np.array(real)
    neq = np.not_equal(pred, real)
    # err = float(neq.sum())/pred.shape[0]
    # fpr = float(np.logical_and(pred==1,neq).sum())/(real==0).sum()
    fnr = np.logical_and(pred == 0, neq).sum() / (real == 1).sum() if (real == 1).sum() > 0 else 0.0
    # 将无法计算fnr的值从0改为0.0,保证在train和inference调用生成str_logs时不会引起Precision not allowed in integer format specifier的报错
    balanced_acc = balanced_accuracy_score(real, pred)
    recall = recall_score(real, pred, average='weighted')
    metrics_meters = {'acc': balanced_acc, 'recall': recall, 'fnr': fnr}

    return metrics_meters


def group_argtopk(groups, data, k=1):
    order = np.lexsort((data, groups))
    groups = groups[order]
    data = data[order]
    index = np.empty(len(groups), 'bool')
    index[-k:] = True
    index[:-k] = groups[k:] != groups[:-k]
    return list(order[index])


def group_max(groups, data, nmax):
    out = np.empty(nmax)
    out[:] = np.nan
    order = np.lexsort((data, groups))
    groups = groups[order]
    data = data[order]
    index = np.empty(len(groups), 'bool')
    index[-1] = True
    index[:-1] = groups[1:] != groups[:-1]
    out[groups[index]] = data[index]
    return out


class MILdataset(data.Dataset):
    def __init__(self, libraryfile='', transform=None):
        lib = torch.load(libraryfile)
        slides = []
        patch_size = []
        for i, name in enumerate(lib['slides']):
            sys.stdout.write('==>Opening SVS headers: [{}/{}]\r'.format(i + 1, len(lib['slides'])))
            sys.stdout.flush()
            slides.append(openslide.OpenSlide(name))
            patch_size.append(int(lib['patch_size'][i]))
        print('')
        grid = []
        slideIDX = []
        for i, g in enumerate(lib['grid']):
            grid.extend(g)
            slideIDX.extend([i] * len(g))

        print('Number of tiles: {}'.format(len(grid)))
        self.slidenames = lib['slides']
        self.slides = slides
        self.targets = lib['targets']
        self.grid = grid
        self.slideIDX = slideIDX
        self.transform = transform
        self.mode = None
        self.size = patch_size
        self.level = lib['level']
    def setmode(self, mode):
        self.mode = mode

    def maketraindata(self, idxs):
        self.t_data = [(self.slideIDX[x], self.grid[x], self.targets[self.slideIDX[x]]) for x in idxs]
    def shuffletraindata(self):
        self.t_data = random.sample(self.t_data, len(self.t_data))

    def __getitem__(self, index):
        if self.mode == 1:
            slideIDX = self.slideIDX[index]
            coord = self.grid[index]
            (x, y) = coord
            img = self.slides[slideIDX].read_region(coord, self.level,
                                                    (self.size[slideIDX], self.size[slideIDX])).convert('RGB')

            if img.size != (256, 256):
                img = img.resize((256, 256), Image.BILINEAR)

            if self.transform is not None:
                img = self.transform(img)

            return img

        elif self.mode == 2:
            slideIDX, coord, target = self.t_data[index]
            (x, y) = coord
            img = self.slides[slideIDX].read_region(coord, self.level,
                                                    (self.size[slideIDX], self.size[slideIDX])).convert('RGB')


            if img.size != (256, 256):
                img = img.resize((256, 256), Image.BILINEAR)
            if self.transform is not None:
                img = self.transform(img)

            return img, target

    def __len__(self):
        if self.mode == 1:
            return len(self.grid)
        elif self.mode == 2:
            return len(self.t_data)


class LinearClassifier(nn.Module):
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)
        return self.linear(x)


if __name__ == '__main__':
    main()
