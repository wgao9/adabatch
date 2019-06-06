import argparse
import os
import sys
import time

from utee import misc
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import dataset
import model
import numpy as np
import math

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--wd', type=float, default=0.0001, help='weight decay')
parser.add_argument('--min_batch_size', type=int, default=2, help='minimum input batch size for training (default: 4)')
parser.add_argument('--init_batch_size', type=int, default=16, help='initial input batch size for training')
parser.add_argument('--max_batch_size', type=int, default=512, help='maximum input batch size for training (default: 1024)')
parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 1e-3)')
parser.add_argument('--gpu', default=None, help='index of gpus to use')
parser.add_argument('--ngpu', type=int, default=6, help='number of gpus to use')
parser.add_argument('--seed', type=int, default=117, help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=100,  help='how many batches to wait before logging training status')
parser.add_argument('--test_interval', type=int, default=1,  help='how many epochs to wait before another test')
parser.add_argument('--logdir', default='log/default', help='folder to save to the log')
parser.add_argument('--data_root', default='/tmp/public_dataset/pytorch/', help='folder to save the model')
parser.add_argument('--decreasing_lr', default='80,120', help='decreasing strategy')
parser.add_argument('--update_bs_interval', type=int, default=101, help='how many batches to update batch size')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--beta', type=float, default=0.9, help='beta for moving average')
parser.add_argument('--target_acc', type=float, default=98.0, help='target accuracy')
parser.add_argument('--smoothing_weight', type=float, default=0.8, help='smoothing weight for target accuracy')

parser.add_argument('--model', type=str, default='3_layer', help='model: 3_layer/logistic')
parser.add_argument('--schedule', type=str, default='fix', help='batch size schedule: fix/multi/de/ours')
parser.add_argument('--multi', type=float, default=1.0, help='multiplicative for batch size increasing')
parser.add_argument('--increasing', type=bool, default=False, help='whether to enforce increasing bs')
args = parser.parse_args()
args.logdir = os.path.join(os.path.dirname(__file__), args.logdir)
misc.logger.init(args.logdir, 'train_log')
print = misc.logger.info

# select gpu
args.gpu = misc.auto_select_gpu(utility_bound=0, num_gpu=args.ngpu, selected_gpus=args.gpu)
args.ngpu = len(args.gpu)

# logger
misc.ensure_dir(args.logdir)
print("=================FLAGS==================")
for k, v in args.__dict__.items():
    print('{}: {}'.format(k, v))
print("========================================")

# seed
args.cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

#compute the statistic for updating batchsize
def get_stats(lr, m, v, loss):
    if args.schedule in ['fix','multi']:
        return 1.0
    elif args.schedule == 'step':
        return lr**2
    elif args.schedule == 'de':
        var, l_2 = 0.0, 0.0
        for t in range(len(m)):
            var += torch.sum(v[t] - m[t]**2).item()
            l_2 += torch.sum(v[t]).item()
        return (lr*var/l_2)**2
    elif args.schedule == 'cabs':
        var = 0.0
        for t in range(len(m)):
            var += torch.sum(v[t] - m[t]**2).item()
        return (lr*var/loss)**2
    #our algorithm: lr**2*(v-m**2)
    elif args.schedule == 'ours':
        var = 0.0
        for t in range(len(m)):
            var += torch.sum(v[t] - m[t]**2).item()
        return (lr**2)*var
    else:
        raise NotImplementedError
    
def update_bs(batch_size, stats_new, stats_old):
    if stats_new <= 1e-15 or stats_old <= 1e-15:
        return batch_size
    new_bs = batch_size*args.multi*np.sqrt(stats_new/stats_old)
    if args.increasing:
        return math.ceil(np.clip(new_bs, batch_size, args.max_batch_size))
    else:
        return math.ceil(np.clip(new_bs, args.min_batch_size, args.max_batch_size))

# data loader

# model
if args.model == 'logistic':
    model = model.mnist(input_dims=784, n_hiddens=[], n_class=10)
else:
    model = model.mnist(input_dims=784, n_hiddens=[256,256], n_class=10)
model = torch.nn.DataParallel(model, device_ids= range(args.ngpu))
if args.cuda:
    model.cuda()

# optimizer; SGD with momentum=0.9
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
best_acc, old_file = 0, None
t_begin = time.time()
adap_batch_size = args.init_batch_size
m, v = {}, {}
steps, datas = 0, 0
stats = 0.0
perf_inf = "#"
for arg in sys.argv:
    perf_inf += arg+"\t"
perf_inf += "\n"
perf_inf = "#steps\tcurrent_bs\tdata\tloss\taccuracy\n"
average_acc = 10
average_loss = 0.5
smoothed_loss = 0.0

try:
    # ready to go
    for epoch in range(args.epochs):
        model.train()

        #partite the data by batch_size=adap_batch_size
        train_loader, test_loader = dataset.get(batch_size=adap_batch_size, data_root=args.data_root, num_workers = 1)

        #train 100 steps
        for batch_idx, (data, target) in enumerate(train_loader):
            indx_target = target.clone()
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)

            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
 
            #computing moving average of first moment(m) and second moment(v) of gradients
            params = list(model.parameters())
            for ix in range(len(params)):
                if ix not in m.keys():
                    m[ix] = np.sqrt(adap_batch_size) * params[ix].grad.data.clone()
                    v[ix] = adap_batch_size * params[ix].grad.data ** 2
                else:
                    m[ix] = args.beta*m[ix] + (1-args.beta)*np.sqrt(adap_batch_size)*params[ix].grad.data
                    v[ix] = args.beta*v[ix] + (1-args.beta)*adap_batch_size*params[ix].grad.data ** 2
    
            smoothed_loss = args.beta*smoothed_loss + (1-args.beta)*loss.item()
            optimizer.step()
            
            #compute total steps and total datas
            if (batch_idx+1) % args.update_bs_interval == 0 and batch_idx > 0:
                steps += (batch_idx+1)
                datas += (batch_idx+1) * len(data)
                break

        #compute model accuracy for validation dataset per 100 steps
        model.eval()
        loss, correct = 0, 0
        for idx, (data, target) in enumerate(test_loader):
            indx_target = target.clone()
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            with torch.no_grad():
                data, target = Variable(data, volatile=True), Variable(target)
                output = model(data)
                loss += F.cross_entropy(output, target).data.item()
                pred = output.data.max(1)[1]  # get the index of the max log-probability
                correct += pred.cpu().eq(indx_target).sum()
        loss = loss / len(test_loader) # average over number of mini-batch
        acc = 100.0 * correct.item()/ len(test_loader.dataset)

        #smoothed accuracy = w*(smoothed accuracy last time)+(1-w)*(current accuracy)
#        average_acc_6 = 0.6*average_acc_6 + i0.4*acc
        average_loss = args.smoothing_weight*average_loss + (1-args.smoothing_weight)*loss
        average_acc = args.smoothing_weight*average_acc + (1-args.smoothing_weight)*acc
        print('steps: {}, batchsize: {}, data: {}, loss: {:.4f}/{:.4f}, Accuracy  {:.2f}%/{:.2f}%'.format(
                steps, adap_batch_size, datas, loss, average_loss, acc, average_acc))
        perf_inf += "%d\t"%steps
        perf_inf += "%d\t"%adap_batch_size
        perf_inf += "%d\t"%datas
        perf_inf += "%.4f\t"%average_loss
        perf_inf += "%.2f\n"%average_acc

        #if smoothed accuracy > target, stop training
        #if average_acc_6 > args.target_acc:
        #    print('Trainingpauses, steps: {}, average_batchsize: {:.2f}, data: {}'.format(steps, datas/steps, datas))
        #    average_acc_6 = 0
        if average_acc > args.target_acc:
            print('Training ends, steps: {}, average_batchsize: {:.2f}, data: {}'.format(steps, datas/steps, datas))
            break

        #update batch size
        stats_new = get_stats(args.lr, m, v, smoothed_loss)
        print("stats_old = {:.4f}, stats_new = {:.4f}".format(stats, stats_new))
        adap_batch_size = update_bs(adap_batch_size, stats_new, stats)
        stats = stats_new        

except Exception as e:
    import traceback
    traceback.print_exc()
finally:
    filename = "mnist_"+args.model+"_"+args.schedule+"_"+str(args.init_batch_size)+".dat"
    pathname = "mnist/"+args.model
    if not os.path.exists(pathname):
        os.makedirs(pathname)
    filepath = os.path.join(pathname, filename)

    with open(filepath, "w") as f:
        f.write(perf_inf)
    print("Total Elapse: {:.2f}, Best Result: {:.3f}%".format(time.time()-t_begin, best_acc))


