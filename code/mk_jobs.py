import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--clip", dest="clip", type=str,
    help="name of clip to calculcate FC data from (e.g., MOVIE1, bridgeville)")
parser.add_argument("--behav", dest="behav", type=str,
    help="name of behavior you are trying to predict (e.g., ListSort_Unadj)")
parser.add_argument("--gsr", dest="gsr", type=int)
parser.add_argument("--zscore", dest="zscore", type=int)
parser.add_argument("--model_type", dest="model_type", type=str, default="cpm",
                    help="type of model to use (cpm or krr)")
parser.add_argument("--total_trs", dest="total_trs", type=int, default=None)
parser.add_argument("--cut_out_rest", dest="cut_out_rest", type=int, default=1,
                    help="whether or not to cut out rest blocks when calculating FC (only applies to full movie runs)")
parser.add_argument("--same_rest_block", dest="same_rest_block", type=int, default=0,
                    help="whether to use the matched block from the REST run in the same session (only applies to individual clips)")
parser.add_argument("--n_iter", dest="n_iter", type=int, default=0,
                    help="how many iterations (lines) to write")
parser.add_argument("--rand_behav", dest="rand_behav", type=int, default=0,
                    help="whether or not to randomize behavior (for perm test)")
parser.add_argument("--reg_cfds", dest="reg_cfds", type=int, default=1,
                    help="whether or not to regress confounds (motion, TOD) from y")
parser.add_argument("--f_name", dest="f_name", type=str, default="jobs.txt", help="what to call jobs file")
parser.add_argument("--rewrite", dest="rewrite", type=bool, default=False, help="whether to rewrite jobs file if it already exists, or append to it")
args = parser.parse_args()

clip = args.clip
behav = args.behav
gsr = args.gsr
zscore = args.zscore
model_type = args.model_type
n_iter = args.n_iter
rand_behav = args.rand_behav
total_trs = args.total_trs
cut_out_rest = args.cut_out_rest
same_rest_block = args.same_rest_block
reg_cfds = args.reg_cfds
f_name = args.f_name
rewrite = args.rewrite

if rewrite:
    if os.path.exists(f_name):
        os.remove(f_name)
        print("removing existing jobs file")

for i in range(n_iter):
    with open(f_name, 'a+') as f:
        l = ''.join(["python cpm_wrapper.py --clip ", clip,
        " --behav ",behav,
        " --gsr ",str(gsr),
        " --zscore ",str(zscore),
        " --model_type ",str(model_type),
        " --rand_behav ", str(rand_behav),
        " --iter_no ",str(i)])
        if total_trs is not None:
            l = ''.join([l, " --total_trs ", str(total_trs)])
        if reg_cfds ==0:
            l = ''.join([l, " --reg_cfds ", str(reg_cfds)])
        if cut_out_rest ==0:
            l = ''.join([l, " --cut_out_rest ", str(cut_out_rest)])
        if same_rest_block ==1:
            l = ''.join([l, " --same_rest_block ", str(same_rest_block)])
        f.write(l + '\n')
