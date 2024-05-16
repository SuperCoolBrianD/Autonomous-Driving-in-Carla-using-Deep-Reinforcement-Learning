# Analyze trained results and produce plots and reports

import os
from datetime import datetime
import pickle
import numpy as np


def get_all_training_folders(parent_folder: str):
    folders = os.listdir(parent_folder)
    folders = [f for f in folders if os.path.isdir(f)]
    folders = [f for f in folders if f.startswith('2024')]
    folders.sort()
    return folders


def reform_logs(logs: dict) -> (dict, dict):
    train_log = {  # each value is one epoch
        'reward': np.array(logs['reward']),
        'step_count': np.array(logs['step_count']),
        'lr': np.array(logs['lr']),
    }
    eval_log = {  # interval is manually defined
        'reward': np.array(logs['eval reward']),
        'step_count': np.array(logs['eval step_count']),
        'reward_sum': np.array(logs['eval reward (sum)']),
    }
    return train_log, eval_log


def make_graph(train_log, eval_log, dt_stamp:str, text_anno: str):
    import matplotlib.pyplot as plt
    import numpy as np

    # number of samples
    n_sample = len(train_log['reward'])
    n_eval = len(eval_log['reward'])
    sample_per_eval = round(n_sample / n_eval)  # 10 by default
    print(f"  This training started at {dt}. Total {n_sample} collected samples, "
          f"evaluated every {n_eval} epochs.")

    # plot the logs
    plt.figure(figsize=(16, 10))
    plt.subplot(2, 2, 1)
    plt.plot(np.arange(0, n_sample), train_log['reward'], label='train reward')
    plt.plot(np.arange(0, n_eval) * sample_per_eval, eval_log['reward'], label='eval reward')
    plt.xlabel('n_sample')
    plt.ylabel('reward')
    plt.legend()
    plt.grid()

    plt.subplot(2, 2, 3)
    plt.plot(np.arange(0, n_sample), train_log['step_count'], label='train step count')
    plt.plot(np.arange(0, n_eval) * sample_per_eval, eval_log['step_count'], label='eval step count')
    plt.xlabel('n_sample')
    plt.ylabel('step_count')
    plt.legend()
    plt.grid()

    # plot learning rate
    plt.subplot(2, 2, 2)
    plt.plot(np.arange(0, n_sample), train_log['lr'], label='lr')
    plt.xlabel('n_sample')
    plt.ylabel('learning rate')
    plt.legend()
    plt.grid()

    plt.text(0.02, 0.96, f"Training started at {dt}", fontsize=12, transform=plt.gcf().transFigure)
    plt.text(0.02, 0.93, f"Graph generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
             fontsize=12, transform=plt.gcf().transFigure)
    plt.text(0.55, 0.4, text_anno, fontsize=12, transform=plt.gcf().transFigure)

    # save graphs to file
    if not os.path.exists('plots'):
        plt.show()
    else:
        plt.savefig(f'plots/results-{dt_stamp}.png')


if __name__ == '__main__':
    result_folder = '.'
    log_folders = get_all_training_folders(result_folder)
    print(f"Found {len(log_folders)} training folders.")
    for folder in log_folders:
        dt = datetime.strptime(folder, '%Y%m%d-%H%M%S')  # folder name is time string
        print(f"Processing folder {folder}")

        # get the latest log file
        log_file = os.path.join(result_folder, folder, 'logs.pkl')
        with open(log_file, 'rb') as f:
            logs = pickle.load(f)

        # reform logs
        train_log, eval_log = reform_logs(logs)
        try:
            with open(os.path.join(result_folder, folder, 'note.txt'), 'r') as f:
                text = f.read()
        except:
            text = 'No other notes.'

        # make graph
        make_graph(train_log, eval_log, folder, text)

