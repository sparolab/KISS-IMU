import argparse, ast

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--result-dir', type=str, required=True)

    parser.add_argument('--data-type',    type=str, required=True)
    parser.add_argument('--data-root',    type=str, required=True)
    parser.add_argument('--train-seqs',   nargs='+', type=str)
    parser.add_argument('--valid-seqs',   nargs='+', type=str)
    parser.add_argument('--inference-seqs',   nargs='+', type=str)

    parser.add_argument('--train-name',        type=str)
    parser.add_argument('--pretrained-model',  type=str, default=None)

    parser.add_argument('--batch-size', type=int,   default=10)
    parser.add_argument('--epoch',      type=int,   default=30)
    parser.add_argument('--worker-num', type=int,   default=2)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--device',     type=str,   default='cuda:0')

    parser.add_argument('--lm-weight',  type=lambda x: list(ast.literal_eval(x)), 
                                        default='(1,0.1,5,0.1,0.1)')
    parser.add_argument('--rot-w',      type=float, default=1e3)
    parser.add_argument('--vel-w',      type=float, default=1e1)
    parser.add_argument('--pos-w',      type=float, default=1e2)
    
    parser.add_argument('--cov-r-w',      type=float, default=1e-4)
    parser.add_argument('--cov-v-w',      type=float, default=1e-4)
    parser.add_argument('--cov-t-w',      type=float, default=1e-4)

    parser.add_argument('--rot-cov-scaler', type=float)
    parser.add_argument('--vel-cov-scaler', type=float)
    parser.add_argument('--pos-cov-scaler', type=float)

    parser.add_argument('--lo-model',     type=str, default='fast_gicp')
    parser.add_argument('--prop-cov',     action='store_false', help='default: False')
    parser.add_argument('--train-ratio',  type=float, default=1.0)
    parser.add_argument('--gmm-comp-num', type=int, default=0)
    
    parser.add_argument('--scheduler',         type=str,   default='CosineAnnealingLR', choices=['ReduceLROnPlateau', 'StepLR', 'CosineAnnealingLR'], help='scheduler type (default: ReduceLROnPlateau)')
    parser.add_argument('--scheduler-patience', type=int,  default=8,             help='patience for ReduceLROnPlateau (default: 8)')
    parser.add_argument('--scheduler-factor',  type=float, default=0.8,           help='factor for learning rate reduction (default: 0.8)')
    parser.add_argument('--scheduler-step-size', type=int, default=5,             help='step size for StepLR (default: 5)')
    parser.add_argument('--scheduler-min-lr',  type=float, default=1e-7,          help='minimum learning rate (default: 1e-7)')

    # Adaptive weight control
    parser.add_argument('--use-adaptive-weight', action='store_true', help='Use adaptive weight during inference')
    parser.add_argument('--no-adaptive-weight', action='store_true', help='Do not use adaptive weight during inference')
    
    parser.add_argument('--use-submap', action='store_true', help='Use adaptive weight during inference')
    parser.add_argument('--no-submap', action='store_true', help='Do not use adaptive weight during inference')

    # Supervision source: by default labels come from the better of ICP / PGO
    # (self-supervised pseudo-label). With --use-gt, labels are taken directly
    # from `sample['gt_pose1']` — useful for an oracle / ablation run that
    # isolates the contribution of GMM reweighting from the LO pseudo-label.
    parser.add_argument('--use-gt', action='store_true',
                        help='Train against GT poses instead of the ICP/PGO pseudo-label')

    args = parser.parse_args()
    return args
