import os
import json
import argparse
import numpy as np

splits = ['training', 'testing']
infraction_types = ['collisions_pedestrian', 'collisions_vehicle', 'collisions_layout', 'red_light', 'stop_infraction', 'route_dev', 'route_timeout', 'vehicle_blocked', 'outside_route_lanes']
penalties = ['.5x', '.6x', '.65x', '.7x', '.8x', 'STOP', 'STOP', 'STOP', '']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repetitions', type=int, default=10)
    parser.add_argument('--agent', type=str, default='image_agent')
    parser.add_argument('--split', type=str, default='testing', choices=['devtest','testing','training','debug'])
    parser.add_argument('--log_path', type=str, default='leaderboard/data')
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    return args

def get_metrics(metrics, metric_type, routes):
        return [metrics[route][metric_type] for route in routes]

def plot_metrics(args, metrics, routes):
    import seaborn as sns
    sns.set()
    colors = sns.color_palette("Paired")
    import matplotlib.pyplot as plt

    # time to plot
    fig = plt.gcf()
    fig.set_size_inches(12,8)
    save_path_base = os.path.join(args.log_path, f'logs_rep{args.repetitions}', args.agent, f'{args.split}_plots')
    if not os.path.exists(save_path_base):
        os.makedirs(save_path_base)


    # infraction metrics
    W = 3
    plot_labels = [infraction.replace('_', '\n') for infraction in infraction_types]
    plot_labels = [f'{label}\n{penalty}' for label, penalty in zip(plot_labels, penalties)]
    x_plot = np.arange(len(plot_labels))*2*W
    for route in routes:

        means = [metrics[route][f'{infraction} mean'] for infraction in infraction_types]
        stds = [metrics[route][f'{infraction} std'] for infraction in infraction_types]
        maxs = [metrics[route][f'{infraction} max'] for infraction in infraction_types]
        mins = [metrics[route][f'{infraction} min'] for infraction in infraction_types]

        plt.bar(x_plot, means, alpha=0.75, width=W)
        plt.errorbar(x_plot, means, yerr=stds, fmt="ok", capsize=3, alpha=0.75, lw=1, ms=0)
        plt.scatter(x_plot, maxs, s=5, edgecolors='black', alpha=0.5)
        plt.scatter(x_plot, mins, s=5, edgecolors='black', alpha=0.75)
        plt.xticks(x_plot, plot_labels, fontsize=8)
        plt.yticks(np.arange(max(maxs) + 1))
        plt.ylim(-0.5, max(maxs) + 0.5)
        plt.title(f'{route} average infractions')
        plt.xlabel('infraction type')
        plt.ylabel('# of infractions')
        save_path = os.path.join(save_path_base, f'{route}_infractions.png')
        driving_score = metrics[route]['driving score mean']
        rcompletion_score = metrics[route]['route completion mean']
        plt.text(2*W*6, max(maxs)+0.5, 'route completion\ndriving score')
        plt.text(2*W*7.75, max(maxs)+0.5, f'{rcompletion_score:.2f}\n{driving_score:.2f}')
        plt.tight_layout()
        plt.savefig(save_path, dpi=100)
        plt.clf()

    # driving and route completion metrics
    if args.split == 'training':
        split_idx = len(routes)//2
        routes_iter = [routes[:split_idx], routes[split_idx:]]
    elif args.split == 'testing':
        routes_iter = [routes]

    for i, routes in enumerate(routes_iter):
        X = np.arange(len(routes))*W
        
        plots = []
        plot_labels = ['route completion', 'driving score']
        for j, t in enumerate(plot_labels):
            color = colors[2*j]
            x_plot = 3*X+j*W
            means = get_metrics(metrics, f'{t} mean', routes)
            stds = get_metrics(metrics, f'{t} std', routes)
            barplot = plt.bar(x_plot, means, color=color, width=W, alpha=0.75)
            plt.errorbar(x_plot, means, yerr=stds, fmt="ok", capsize=3, alpha=0.75, color=color, lw=1, ms=0)
            maxs = get_metrics(metrics, f'{t} max', routes)
            mins = get_metrics(metrics, f'{t} min', routes)
            plt.scatter(x_plot, maxs, color=color, s=5, edgecolors='black', alpha=0.5)
            plt.scatter(x_plot, mins, color=color, s=5, edgecolors='black', alpha=0.75)
            plots.append(barplot)

        plt.legend(plots, plot_labels)

        numbers = [route[-2:] for route in routes]
        plt.xticks(3*X+W/2, numbers, fontsize=12)
        plt.xlabel('route #')
        plt.ylabel('score')
        plt.title(f'avg driving/route scores')
        plt.ylim(-5,105)
        save_path = os.path.join(save_path_base, f'overall_score_metrics_{i}.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=100)
        plt.clf()


def main(args):

    log_path = os.path.join(args.log_path, f'logs_rep{args.repetitions}', args.agent, args.split)
    log_fnames = sorted([os.path.join(log_path, fname) for fname in os.listdir(log_path) if fname.endswith('txt')])

    metrics = {}
    routes = []

    for fname in log_fnames:
        with open(fname) as f:
            log = json.load(f)
        route = fname.split('/')[-1].split('.')[0]
        routes.append(route)
        metrics[route] = {}
        records = log['_checkpoint']['records']

        # driving score, route completion score metrics
        dscores = [record['scores']['score_composed'] for record in records]
        rcscores = [record['scores']['score_route'] for record in records]
        metrics[route]['driving score mean'] = np.mean(dscores)
        metrics[route]['driving score std'] = np.std(dscores)
        metrics[route]['driving score max'] = np.amax(dscores)
        metrics[route]['driving score min'] = np.amin(dscores)
        metrics[route]['route completion mean'] = np.mean(rcscores)
        metrics[route]['route completion std'] = np.std(rcscores)
        metrics[route]['route completion max'] = np.amax(rcscores)
        metrics[route]['route completion min'] = np.amin(rcscores)

        # infractions
        for inf_type in infraction_types:
            num_infractions = [len(record['infractions'][inf_type]) for record in records]
            metrics[route][f'{inf_type} mean'] = np.mean(num_infractions)
            metrics[route][f'{inf_type} std'] = np.std(num_infractions)
            metrics[route][f'{inf_type} max'] = np.amax(num_infractions)
            metrics[route][f'{inf_type} min'] = np.amin(num_infractions)

    if args.plot:
        plot_metrics(args, metrics, routes)

    overall_dscore_mean = np.mean([metrics[route]['driving score mean'] for route in routes])
    overall_rcscore_mean = np.mean([metrics[route]['route completion mean'] for route in routes])
    print(f'On the {args.split} routes over {args.repetitions} repetitions:')
    print(f'avg driving score \t = {overall_dscore_mean:.2f}')
    print(f'avg route completion \t = {overall_rcscore_mean:.2f}')
        
if __name__ == '__main__':
    args = parse_args()
    main(args)
