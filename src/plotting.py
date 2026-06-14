import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sksurv.nonparametric import kaplan_meier_estimator

def plot_kaplan_meier(dataset, method, sgs=None, colors=None):
    if sgs is None:
        sgs = [0, 1, 3, 4]
    if colors is None:
        colors = [
            'limegreen', 'dodgerblue', 'violet', 'tomato', 'chocolate', 'blue', 'green', 'red', 'orange'
        ]

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    line_width = 3
    fs = 25

    plt.figure(figsize=(6, 4))
    label_added = False

    for sg_i, color in zip(sgs, colors):
        sg = method.subgroups[sg_i]
        print(color, '-- size:', sum(sg), '-- rule:\n', method.rules[sg_i], "\n")
        x, y, conf_int = kaplan_meier_estimator(dataset.event[sg], dataset.time[sg], conf_type="log-log")
        plt.step(x, y, where="post", label='Subgroup ' + str(sg_i) if label_added else '', color=color, linewidth=line_width)
        plt.fill_between(x, conf_int[0], conf_int[1], color=color, alpha=0.15, step="post")

    x, y, conf_int = kaplan_meier_estimator(dataset.event, dataset.time, conf_type="log-log")
    plt.step(x, y, where="post", label='Population', color='k', linewidth=line_width)
    plt.fill_between(x, conf_int[0], conf_int[1], color='k', alpha=0.05, step="post")

    plt.xlabel('Follow-up time (months)', alpha=1, fontsize=fs, labelpad=40)
    plt.xticks(fontsize=fs)
    plt.ylim(0, 1.01)
    plt.xlim(0, max(x))
    plt.yticks(fontsize=fs)
    plt.subplots_adjust(left=0.1, bottom=0.1)
    plt.ylabel('Survival probability', alpha=1, fontsize=fs, labelpad=40)
    plt.title('')
    
    for spine in ['top', 'right']:
        plt.gca().spines[spine].set_visible(False)
    plt.gca().spines['left'].set_color('black')
    plt.gca().spines['bottom'].set_color('black')
    plt.gca().spines['bottom'].set_position(('outward', 14))
    plt.gca().spines['left'].set_position(('outward', 14))
    plt.gca().spines['left'].set_linewidth(line_width)
    plt.gca().spines['bottom'].set_linewidth(line_width)
    plt.tick_params(axis='both', which='major', direction='out', length=14, width=line_width/2, colors='black', labelsize=fs)
    plt.tick_params(axis='both', which='minor', direction='out', length=14, width=line_width / 2, colors='black', labelsize=fs)
    plt.grid(False)
    
    legend = plt.legend(
        bbox_to_anchor=(0.75, 0), frameon=True, edgecolor='black', facecolor='white', 
        framealpha=1.0, fontsize=fs, loc='lower center', mode='fit', ncols=3, 
        handlelength=1, fancybox=False
    )
    legend.get_frame().set_linewidth(line_width)
    plt.show()

def plot_validation(scores, mu, sigma, exceptionalities, p_values, corrected_alpha, cutoff):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['mathtext.fontset'] = 'stix'
    x = np.linspace(mu - 6*sigma, mu + 6*sigma, 100)
    y = norm.pdf(x, mu, sigma)
    plt.figure(figsize=(8, 5))
    plt.title('Histogram vs CLT-parameterized Normal Distribution')
    plt.xlabel('Value')
    plt.ylabel('Probability Density')
    plt.grid(axis='y', alpha=0.5)
    plt.hist(scores, bins=min(len(scores), 30), density=True, alpha=0.6, color='skyblue', label='Sample data')
    plt.plot(x, y, label=f'$\\mathcal{{N}}$($\\mu$ = {mu:.3}, $\\sigma$ = {sigma:.3})', color='darkorange', linewidth=2)
    plt.axvline(mu, color='darkorange', linestyle='--')
    
    colors = ['limegreen', 'dodgerblue', 'violet', 'tomato', 'chocolate', 'blue', 'green', 'red', 'orange']
    for i, (exc, p_val) in enumerate(zip(exceptionalities, p_values)):
        color = colors[i % len(colors)]
        plt.axvline(exc, color=color, linestyle='--', label=f'Subgroup {i+1} (p = {p_val:.3})')
        
    plt.axvline(cutoff, color='lightgrey', linestyle='--', label=f'Cutoff (corrected $\\alpha$ = {corrected_alpha:.3})')
    plt.legend()
    plt.show()