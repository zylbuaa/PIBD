import argparse

def _process_args():
    r"""
    Function creates a namespace to read terminal-based arguments for running the experiment

    Args
        - None 

    Return:
        - args : argparse.Namespace

    """

    parser = argparse.ArgumentParser(description='Configurations for SurvPath Survival Prediction Training')

    #---> study related
    parser.add_argument('--study', type=str, default='tcga_stad',help='study name')
    parser.add_argument('--task', type=str, choices=['survival'],default='survival', help='task name')
    parser.add_argument('--n_classes', type=int, default=4, help='number of classes (4 bins for survival)')
    parser.add_argument('--results_dir', default="/home/zhany0x/Documents/experiment/PIBD-new/", help='results directory (default: ./results)')
    # parser.add_argument('--specific_simple', default="seed1", help='specific simple name')
    parser.add_argument("--type_of_path", type=str, default="combine", choices=["xena", "hallmarks", "combine"])
    parser.add_argument('--mode', type=str, default="swin", choices=["cluster","resnet50","swin"], help = "DeepMISL using cluster mode")

    #----> data related
    parser.add_argument('--data_root_dir', type=str, default="/home/zhany0x/Documents/data/ctranspath-pibd/stad/tiles-l1-s224/feats-l1-s224-CTransPath-sampler/", help='data directory')
    parser.add_argument('--label_file', type=str, default="./datasets_csv/metadata/tcga_stad.csv", help='Path to csv with labels')
    parser.add_argument('--omics_dir', type=str, default="./datasets_csv/raw_rna_data/combine/stad/", help='Path to dir with omics csv for all modalities')
    parser.add_argument('--num_patches', type=int, default=4096, help='number of patches')
    parser.add_argument('--label_col', type=str, default="survival_months_dss", help='type of survival (OS, DSS, PFI)')

    #----> split related 
    parser.add_argument('--k', type=int, default=5, help='number of folds (default: 10)')
    parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
    parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
    parser.add_argument('--split_dir', type=str, default='splits', help='manually specify the set of splits to use, '
                    +'instead of infering from the task and label_frac argument (default: None)')
    parser.add_argument('--which_splits', type=str, default="5foldcv", help='where are splits')

    #----> training related 
    parser.add_argument('--max_epochs', type=int, default=30, help='maximum number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate (default: 0.0001)')
    parser.add_argument('--seed', type=int, default=1, help='random seed for reproducible experiment (default: 1)')
    parser.add_argument('--opt', type=str, default="adam", help="Optimizer")
    parser.add_argument('--reg_type', type=str, default="None", help="regularization type [None, L1, L2]")
    parser.add_argument('--weighted_sample', action='store_false', default=True, help='enable weighted sampling')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--bag_loss', type=str, choices=["nll_surv", "rank_surv", "cox_surv"], default='nll_surv',
                        help='survival loss function (default: ce)')
    parser.add_argument('--alpha_surv', type=float, default=0.5, help='weight given to uncensored patients')
    parser.add_argument('--reg', type=float, default=1e-3, help='weight decay / L2 (default: 1e-5)')
    parser.add_argument('--max_cindex', type=float, default=0.0, help='maximum c-index')

    #---> model related
    parser.add_argument('--method', type=str, default="PIBD", help='methd type')
    parser.add_argument('--encoding_dim', type=int, default=768, help='WSI encoding dim (1024 for resnet50, 768 for swin)')
    parser.add_argument('--wsi_projection_dim', type=int, default=256, help="projection dim of features")
    parser.add_argument('--omics_format', type=str, default="pathways", choices=["gene","groups","pathways"], help='format of omics data')
    parser.add_argument('--alpha', type=int, default=0.1, help="hyperparameters of PIB loss function")
    parser.add_argument('--beta', type=int, default=0.01, help="hyperparameters of PIB loss function")
    parser.add_argument('--gamma', type=int, default=1, help="hyperparameters of proxy loss function")
    parser.add_argument('--sigma', type=int, default=0.1, help="hyperparameters of PID loss function, lambda in paper")
    parser.add_argument('--ratio_wsi', type=float, default=0.5, help='hyperparameters')
    parser.add_argument('--ratio_omics', type=float, default=0.8, help='hyperparameters')
    parser.add_argument('--sample_num', type=int, default=50, help='hyperparameters')

    #---> gpu id
    parser.add_argument('--gpu', type=str, default="0", help='gpu id')

    #---> only test the model
    parser.add_argument('--only_test', action='store_true', default=False, help='only test')

    args = parser.parse_args()

    if not (args.task == "survival"):
        print("Task and folder does not match")
        exit()

    return args
