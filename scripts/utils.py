import json

def calc_plot_size(params):
    figwidth = min(14, 4 * params['Lx'] / params['Lz'])
    figheight = figwidth * params['Lz'] / params['Lx']
    return figwidth, figheight

def read_params(data_dir="analysis"):
    with open(f'{data_dir}/params.json', 'r') as f:
        params = json.load(f)
    return params

def save_params(params, data_dir="analysis"):
    with open(f'{data_dir}/params.json', 'w') as f:
        json.dump(params, f, indent=2)

def create_params_string(params):
    return "Ra = {:.1E}, Pr = {}, Ek = {}, Theta = {:.3f}".format(params["Ra"], params["Pr"], params["Ek"], params["Theta"])

