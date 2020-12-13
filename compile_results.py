import os
import json
import argparse
import numpy as np

infraction_types = ['collisions_pedestrian', 'collisions_vehicle', 'collisions_layout', 'red_light', 'stop_infraction', 'route_dev', 'route_timeout', 'vehicle_blocked', 'outside_route_lanes']
penalties = ['.5x', '.6x', '.65x', '.7x', '.8x', 'STOP', 'STOP', 'STOP', '']

def get_metrics(metrics, metric_type, routes):
        return [metrics[route][metric_type] for route in routes]

def plot_metrics(args, metrics, routes):
    import seaborn as sns
    sns.set()
    colors = sns.color_palette("Paired")
    import matplotlib.pyplot as plt

    fig = plt.gcf()
    fig.set_size_inches(12,8)

    # infraction metrics - one plot per route
    plot_labels = [infraction.replace('_', '\n') for infraction in infraction_types]
    plot_labels = [f'{label}\n{penalty}' for label, penalty in zip(plot_labels, penalties)]

    # on the bar plot, each infraction type occupes 2 "widths"
    # first width is for the bar showing # of infraction occurences for that type
    # second width is for spacing between infraction types
    W = 3
    x_plot = np.arange(len(plot_labels))*2*W
    for route in routes:

        # retrieve aggregated metrics in a list
        means = [metrics[route][f'{infraction} mean'] for infraction in infraction_types]
        stds = [metrics[route][f'{infraction} std'] for infraction in infraction_types]
        maxs = [metrics[route][f'{infraction} max'] for infraction in infraction_types]
        mins = [metrics[route][f'{infraction} min'] for infraction in infraction_types]

        # plot means as bars, include errorbars and max/mins
        plt.bar(x_plot, means, alpha=0.75, width=W)
        plt.errorbar(x_plot, means, yerr=stds, fmt="ok", capsize=3, alpha=0.75, lw=1, ms=0)
        plt.scatter(x_plot, maxs, s=5, edgecolors='black', alpha=0.5)
        plt.scatter(x_plot, mins, s=5, edgecolors='black', alpha=0.75)

        # resize and label x/y axes, title
        plt.xticks(x_plot, plot_labels, fontsize=8)
        plt.yticks(np.arange(max(maxs) + 1))
        plt.ylim(-0.5, max(maxs) + 0.5)
        plt.title(f'{args.split}/{route} average infractions')
        plt.xlabel('infraction type')
        plt.ylabel('average # of infractions')

        # write the two main score metrics as text
        driving_score = metrics[route]['driving score mean']
        rcompletion_score = metrics[route]['route completion mean']
        plt.text(2*W*6, max(maxs)+0.5, 'route completion\ndriving score')
        plt.text(2*W*7.75, max(maxs)+0.5, f'{rcompletion_score:.2f}\n{driving_score:.2f}')

        plt.tight_layout()
        save_path = os.path.join(args.plot_dir, f'{route}_infractions.png')
        plt.savefig(save_path, dpi=100)
        plt.clf()

    # driving and route completion metrics - one/two plots for the entire split
    if args.split == 'training':
        split_idx = len(routes)//2
        routes_iter = [routes[:split_idx], routes[split_idx:]]
    else:
        routes_iter = [routes]

    for i, routes in enumerate(routes_iter):

        # on the bar plot, each route occupies 3 "widths"
        # first width is route completion bar
        # second width is driving score bar
        # third width is for spacing between routes
        X = np.arange(len(routes))*3*W
        
        plots = []
        plot_labels = ['route completion', 'driving score']
        for j, t in enumerate(plot_labels):
            color = colors[2*j] # arbitrarily chosen seaborn colors
            x_plot = X+j*W # j is an offset for the plot label

            # retrieve aggregated metrics for each route
            means = get_metrics(metrics, f'{t} mean', routes)
            stds = get_metrics(metrics, f'{t} std', routes)
            maxs = get_metrics(metrics, f'{t} max', routes)
            mins = get_metrics(metrics, f'{t} min', routes)

            # plot
            barplot = plt.bar(x_plot, means, color=color, width=W, alpha=0.75)
            plt.errorbar(x_plot, means, yerr=stds, fmt="ok", capsize=3, alpha=0.75, color=color, lw=1, ms=0)
            plt.scatter(x_plot, maxs, color=color, s=5, edgecolors='black', alpha=0.5)
            plt.scatter(x_plot, mins, color=color, s=5, edgecolors='black', alpha=0.75)
            plots.append(barplot) # used for making the legend

        plt.legend(plots, plot_labels)

        # place the route number label right in between and below the two bars
        numbers = [route[-2:] for route in routes]
        plt.xticks(X+W/2, numbers, fontsize=12) 
        plt.xlabel('route #')
        plt.ylabel('average score')
        plt.title(f'average driving/route scores on {args.split} routes')
        plt.ylim(-5,105)

        plt.tight_layout()
        save_path = os.path.join(args.plot_dir, f'overall_score_metrics_{i}.png')
        plt.savefig(save_path, dpi=100)
        plt.clf()


def main(args):

    metrics = {}
    routes = []

    for fname in args.log_fnames:
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

    plot_metrics(args, metrics, routes)

    overall_dscore_mean = np.mean([metrics[route]['driving score mean'] for route in routes])
    overall_rcscore_mean = np.mean([metrics[route]['route completion mean'] for route in routes])
    print(f'On the {args.split} routes over {args.repetitions} repetitions:')
    print(f'avg driving score \t = {overall_dscore_mean:.2f}')
    print(f'avg route completion \t = {overall_rcscore_mean:.2f}')

def parse_args():
    parser = argparse.ArgumentParser()

    # example target_dir: 
    # (compute-1-24) /ssd1/aaronhua/leaderboard/results/image_agent/20201206_2103/testing
    # (compute-1-29) /ssd0/aaronhua/leaderboard/results/image_agent/debug/20201206_2017/devtest
    parser.add_argument('--target_dir', type=str, required=True)
    args = parser.parse_args()

    # augment args with metadata from target_dir
    target_tokens = args.target_dir.split('/')
    if 'debug' in target_tokens:
        insert_strs = ('agent', 'debug', 'date_str', 'split')
    else:
        insert_strs = ('agent', 'date_str', 'split')
    start_token = -len(insert_strs)
    args_dict = vars(args)
    for string, token in zip(insert_strs, target_tokens[start_token:]):
        if string == 'debug':
            args_dict[string] = True
        else:
            args_dict[string] = token

    # get log directory and check for number of repetitions
    log_dir = os.path.join(args.target_dir, 'logs')
    assert len(os.listdir(log_dir)) > 0, 'ERROR: no logs in log directory'
    log_fnames = [os.path.join(log_dir, fname) for fname in os.listdir(log_dir) if fname.endswith('.txt')]
    args_dict['log_fnames'] = sorted(log_fnames)
    with open(log_fnames[0]) as f:
        log = json.load(f)
    args_dict['repetitions'] = len(log['_checkpoint']['records'])

    # construct plot directory
    args_dict['plot_dir'] = os.path.join(args.target_dir, 'plots')
    if not os.path.exists(args.plot_dir):
        os.makedirs(args.plot_dir)

    for key, val in vars(args).items():
        print(f'{key}: {val}')

    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)
