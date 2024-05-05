
#----> general imports
import pandas as pd
import os
from timeit import default_timer as timer
from datasets.dataset_survival import SurvivalDatasetFactory
from utils.core_utils import _train_val
from utils.file_utils import _save_pkl
from utils.general_utils import _get_start_end, _prepare_for_experiment, _reading_experiment_settings
from utils.valid_utils import _val

from utils.process_args import _process_args

import warnings
warnings.filterwarnings("ignore")

def main(args):

    #----> prep for 5 fold cv study
    folds = _get_start_end(args)
    
    #----> storing the val and test cindex for 5 fold cv
    all_val_cindex = []
    all_val_cindex_ipcw = []
    all_val_BS = []
    all_val_IBS = []
    all_val_iauc = []
    all_val_loss = []

    # ----> log
    if args.only_test:
        log_path = os.path.join(args.results_dir, 'log_test.txt')
    else:
        log_path = os.path.join(args.results_dir, 'log_start_{}_end_{}.txt'.format(args.k_start, args.k_end))
    log_file = open(log_path, 'w')

    for i in folds:
        
        datasets = args.dataset_factory.return_splits(
            args,
            csv_path='{}/splits_{}.csv'.format(args.split_dir, i),
            fold=i
        )
        
        print("Created train and val datasets for fold {}".format(i))
        log_file.write("Created train and val datasets for fold {}\n".format(i))

        args.max_cindex = 0.0
        args.max_cindex_epoch = 0


        if args.only_test:
            (val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss) = _val(datasets, i, args, log_file)
        else:
            #----> train and val
            results, (val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss) = _train_val(datasets, i, args,log_file)
            # write results to pkl
            filename = os.path.join(args.results_dir, 'split_{}_results_final.pkl'.format(i))
            print("Saving results...")
            _save_pkl(filename, results)

        all_val_cindex.append(val_cindex)
        all_val_cindex_ipcw.append(val_cindex_ipcw)
        all_val_BS.append(val_BS)
        all_val_IBS.append(val_IBS)
        all_val_iauc.append(val_iauc)
        all_val_loss.append(total_loss.cpu().numpy())


    log_file.close()

    final_df = pd.DataFrame({
        'folds': folds,
        'val_cindex': all_val_cindex,
        'val_cindex_ipcw': all_val_cindex_ipcw,
        'val_IBS': all_val_IBS,
        'val_iauc': all_val_iauc,
        "val_loss": all_val_loss,
        'val_BS': all_val_BS,
    })

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(args.k_start, args.k_end)
    else:
        if args.only_test:
            save_name = 'summary_test.csv'
        else:
            save_name = 'summary.csv'
        
    final_df.to_csv(os.path.join(args.results_dir, save_name))


if __name__ == "__main__":
    # torch.cuda.device_count()

    start = timer()

    #----> read the args
    args = _process_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    #----> prep
    args = _prepare_for_experiment(args)

    #----> create dataset factory
    args.dataset_factory = SurvivalDatasetFactory(
        study=args.study,
        label_file=args.label_file,
        omics_dir=args.omics_dir,
        seed=args.seed, 
        print_info=True, 
        n_bins=args.n_classes,
        label_col=args.label_col, 
        eps=1e-6,
        num_patches=args.num_patches,
        is_mcat = True if args.omics_format == 'groups' else False,
        is_survpath = True if args.omics_format == 'pathways' else False,
        type_of_pathway=args.type_of_path,
        mode = args.mode,
    )

    results = main(args)

    #---> stop timer and print
    end = timer()
    print("finished!")
    print("end script")
    print('Script Time: %f seconds' % (end - start))#----> pytorch imports
