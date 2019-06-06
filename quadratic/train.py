import time
import argparse
import numpy as np
import math
import sys
import os

parser = argparse.ArgumentParser(description='Quadratic Example')
parser.add_argument('--min_bs', type=int, default=1, help='minimum input batch size for training (default: 4)')
parser.add_argument('--init_bs', type=int, default=128, help='initial batch size for training')
parser.add_argument('--max_bs', type=int, default=1024, help='maximum input batch size for training (default: 1024)')
parser.add_argument('--step_size', type=float, default=0.5, help='learning rate (default: 1e-3)')
parser.add_argument('--mu', type=float, default=0.0, help='mean for initial x')
parser.add_argument('--sigma', type=float, default=0.5, help='initial std for noise of gradient')
parser.add_argument('--sigma_rate', type=float, default=0.0, help='decaying rate for std')
parser.add_argument('--epsilon', type=float, default=1e-3, help='goal of loss')
parser.add_argument('--update_interval', type=int, default=1, help='how many batches to update batch size')
parser.add_argument('--print_interval', type=int, default=100, help='how many batches to update batch size')
parser.add_argument('--schedule', type=str, default='fix', help='batch size schedule: fix/multi/step/de/cabs/opti,a;')
parser.add_argument('--multi', type=float, default=1.0, help='multiplicative for batch size increasing')
args = parser.parse_args()

def update_bs(old_bs, s_old, s_new, multi):
    if s_old <= 0 or s_new <= 0:
        return old_bs
    new_bs = old_bs*multi*s_new/s_old
    return math.ceil(np.clip(new_bs, args.min_bs, args.max_bs))

def get_stat(alpha, m, v, f):
    if args.schedule == 'step':
        return alpha
    if args.schedule == 'de':
        return alpha*(v - m**2)/v
    if args.schedule == 'cabs':
        return alpha*(v - m**2)/f
    if args.schedule == 'optimal':
        return alpha*np.sqrt(v - m**2)
    else:
        return 1.0

def func(x):
    return 0.5*(x*x-2*args.mu*x+args.mu*args.mu)

def grad_func(x):
    return x-args.mu

def main():

    perf_inf = "\n"
    for arg in sys.argv:
        perf_inf += arg+" "
    perf_inf += "\n \n"

    for init_bs in range(args.init_bs):
        loss = 1.0
        steps, datas = 0, 0
        batch_size = init_bs+1
        s_old, s_new = 0.0, 0.0

        while loss > args.epsilon:
            steps += 1
            datas += batch_size
            step_size = args.step_size/steps
            sigma = args.sigma*np.power(steps, args.sigma_rate)
        
            #noise_t = noise[steps]/np.sqrt(batch_size)
            #x = x - step_size*(grad_func(x) - noise_t)
            loss = loss*(1-step_size)**2 + step_size**2*sigma**2/batch_size

            if steps%args.update_interval == 0:
                s_old = s_new
                s_new = get_stat(step_size, np.sqrt(2*loss), 2*loss+sigma**2, loss)
                batch_size = update_bs(batch_size, s_old, s_new, args.multi)

            #if steps%args.print_interval == 0:
             #   print("step %d, datas %d, loss %.4f"%(steps, datas, loss))

        print("End of training. Init_bs %d, Steps %d, datas %d, loss %.4f"%(init_bs+1, steps, datas, loss))
        perf_inf += "%d    "%(init_bs+1)
        perf_inf += "%d    "%steps
        perf_inf += "%d    \n"%datas

    filename = args.schedule+".dat"
    pathname = "rate=%.1f"%args.sigma_rate
    if not os.path.exists(pathname):
        os.makedirs(pathname)
    filepath = os.path.join(pathname, filename)

    with open(filepath, "w") as f:
        f.write(perf_inf)
            
if __name__ == "__main__":
    main()
