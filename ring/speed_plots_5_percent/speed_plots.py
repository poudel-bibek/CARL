"""
A consolidated plot for speeds of multiple controllers.
"""

import os 
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_palette(palette='tab10', n_colors = 3)
# set sns white grid with dash lines
sns.set_style("whitegrid", {'grid.linestyle': '--'})

def speed_plot(args, file_dict):

    if args.short_term:
        fontsize = 26
        lw = 3.0
        fig, ax = plt.subplots(figsize=(16, 9), dpi =200)
        legend_labels = [] 
        handles = []
        for category in ['wu','ours_efficiency','ours_safety']:
            current_dir = file_dict[category]
            files = [item for item in os.listdir(current_dir) if item.endswith('.csv')]
            print(f'Category: {category}, Files: {files}')

            avg_speeds_collector = [] 
            for file in files:
                df = pd.read_csv(current_dir + file)

                vehicle_ids = df['id'].unique()
                speeds_total = []
                for vehicle_id in vehicle_ids:
                    speed = df[df['id'] == vehicle_id]['speed'][:13500] # Cut off at 15000
                    speeds_total.append(speed)

                speeds_total = np.array(speeds_total)
                speeds_avg = np.mean(speeds_total, axis=0)
                avg_speeds_collector.append(speeds_avg)

            avg_speeds_collector = np.array(avg_speeds_collector)
            avg_speeds = np.mean(avg_speeds_collector, axis=0)

            std_speeds = np.std(avg_speeds_collector, axis=0)
            print(f'Category: {category}, Avg Speeds: {avg_speeds.shape}, Std Speeds: {std_speeds.shape}')

            if category == 'wu':
                label_text = "Wu"
                
            elif category == 'ours_efficiency':
                label_text = r'Ours $\it{(Efficiency)}$'
                
            elif category == 'ours_safety':
                # write Safety + Stability in italics
                label_text = r'Ours $\it{(Safety)}$'
                

            line, = ax.plot(avg_speeds, label=label_text, linewidth=lw)
            legend_labels.append(label_text)
            handles.append(line)
            ax.fill_between(np.arange(len(avg_speeds)), avg_speeds - std_speeds, avg_speeds + std_speeds, alpha=0.2)
        
        # ax.set_title(f'{self.method_name}: Average speed of vehicles across {self.num_rollouts} rollouts')
        warmup_line = ax.axvline(x=args.warmup, color='black', linestyle='--', dashes=(10, 5), label='Warmup', linewidth=lw)
        handles.append(warmup_line)
        legend_labels.append('Warmup')
        perturb_start_line = ax.axvline(x=args.start_time, color='dodgerblue', linestyle='--', dashes=(10, 5), label='Perturbations start/ end', linewidth=lw)
        handles.append(perturb_start_line)
        legend_labels.append('Perturbations start/ end')
        ax.axvline(x=args.end_time, color='dodgerblue', linestyle='--', dashes=(10, 5), linewidth=lw)

        #ax.set_xlabel('Timesteps', fontsize=fontsize)
        ax.set_ylabel('Average velocity (m/s)', fontsize=fontsize+2)

        # Divide x ticks by 10 and make them seconds
        ax.set_xticks(np.arange(0, 13600, 2000))
        x_labels = [int(item/10) for item in np.arange(0, 13600, 2000)]
        ax.set_xticklabels(x_labels)
        ax.set_xlabel('Time (s)', fontsize=fontsize+2)

        # set xticks and yticks fontsize
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        ax.tick_params(axis='both', which='minor', labelsize=fontsize)
        ax.set_ylim(-0.2,5.5)
        ax.set_xlim(-50,13550)

        # Reorder handles and labels 
        order = [0, 3, 1, 4, 2] # Indices for desired order
        new_handles = [handles[i] for i in order]
        new_labels = [legend_labels[i] for i in order]

        # Place the legend separately at the bottom
        ax.legend(handles=new_handles, labels=new_labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=False, ncol=3, fontsize=fontsize+2)
        fig.tight_layout()
        plt.subplots_adjust(left=0.06, right=0.97)
        plt.savefig('./speeds.png')
     
    else:
        fontsize = 26
        lw = 3.0
        fig, ax = plt.subplots(figsize=(32, 9), dpi =200)
        legend_labels = [] 
        handles = []
        for category in ['wu','ours_efficiency','ours_safety']:
            current_dir = file_dict[category]
            files = [item for item in os.listdir(current_dir) if item.endswith('.csv')]
            print(f'Category: {category}, Files: {files}')

            avg_speeds_collector = [] 
            for file in files:
                df = pd.read_csv(current_dir + file)

                vehicle_ids = df['id'].unique()
                speeds_total = []
                for vehicle_id in vehicle_ids:
                    speed = df[df['id'] == vehicle_id]['speed'][:36000] # Cut off at 15000
                    speeds_total.append(speed)

                speeds_total = np.array(speeds_total)
                speeds_avg = np.mean(speeds_total, axis=0)
                avg_speeds_collector.append(speeds_avg)

            avg_speeds_collector = np.array(avg_speeds_collector)
            avg_speeds = np.mean(avg_speeds_collector, axis=0)

            std_speeds = np.std(avg_speeds_collector, axis=0)
            print(f'Category: {category}, Avg Speeds: {avg_speeds.shape}, Std Speeds: {std_speeds.shape}')

            if category == 'wu':
                label_text = "Wu"
                
            elif category == 'ours_efficiency':
                label_text = r'Ours $\it{(Efficiency)}$'
                
            elif category == 'ours_safety':
                # write Safety + Stability in italics
                label_text = r'Ours $\it{(Safety)}$'
                

            line, = ax.plot(avg_speeds, label=label_text, linewidth=lw)
            legend_labels.append(label_text)
            handles.append(line)
            ax.fill_between(np.arange(len(avg_speeds)), avg_speeds - std_speeds, avg_speeds + std_speeds, alpha=0.2)
        
        # ax.set_title(f'{self.method_name}: Average speed of vehicles across {self.num_rollouts} rollouts')
        warmup_line = ax.axvline(x=args.warmup, color='black', linestyle='--', dashes=(10, 5), label='Warmup', linewidth=lw)
        handles.append(warmup_line)
        legend_labels.append('Warmup')
        perturb_start_line = ax.axvline(x=args.start_time, color='dodgerblue', linestyle='--', dashes=(10, 5), label='Perturbations start/ end', linewidth=lw)
        handles.append(perturb_start_line)
        legend_labels.append('Perturbations start/ end')
        ax.axvline(x=args.end_time, color='dodgerblue', linestyle='--', dashes=(10, 5), linewidth=lw)

        #ax.set_xlabel('Timesteps', fontsize=fontsize)
        ax.set_ylabel('Average velocity (m/s)', fontsize=fontsize+2)

        # Divide x ticks by 10 and make them seconds
        ax.set_xticks(np.arange(0, 36050, 2000))
        x_labels = [int(item/10) for item in np.arange(0, 36050, 2000)]
        ax.set_xticklabels(x_labels)
        ax.set_xlabel('Time (s)', fontsize=fontsize+2)

        # set xticks and yticks fontsize
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        ax.tick_params(axis='both', which='minor', labelsize=fontsize)
        ax.set_ylim(-0.2,5.5)
        ax.set_xlim(-50,36050)

        # Reorder handles and labels 
        order = [0, 1, 2, 3, 4] # Indices for desired order
        new_handles = [handles[i] for i in order]
        new_labels = [legend_labels[i] for i in order]

        # Place the legend separately at the bottom
        ax.legend(handles=new_handles, labels=new_labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=False, ncol=5, fontsize=fontsize+2)
        fig.tight_layout()
        plt.subplots_adjust(left=0.06, right=0.97)
        plt.savefig('./speeds.png')


def main(args):
    """
    Load the files and put them in a dictionary.
    """
    file_dict = {'wu':'./wu/',
                'ours_efficiency':'./efficiency_Ours/',
                 'ours_safety':'./safety_stability_Ours/',}
    speed_plot(args, file_dict)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--horizon', type=int, default=36000) # Short term 15000
    parser.add_argument('--warmup', type=int, default=4000) # Short term 2500

    parser.add_argument('--start_time', type=int, default=10000) # Short term 8000
    parser.add_argument('--end_time', type=int, default=28000) # Warmup + Horizon, Short term 11500
    parser.add_argument('--short_term', action='store_true', default=False)
    args = parser.parse_args()
    main(args)
