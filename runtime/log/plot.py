import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams['hatch.linewidth'] = 0.75
font_size = "large"
font_size_bars = "medium"


def pq_discrete_plot(policy, compression_protocol=None, figsize=(16, 5), legend_loc='lower right', title=""):
    layer_specs, filtered_layers = prepare_plots(policy)

    bar_count = 3
    bar_width = 1 / (bar_count + 0.5)

    fig, p_axs = plt.subplots(figsize=figsize)
    q_axs = p_axs.twinx()
    prune_discrete_subplot(layer_specs, compression_protocol, p_axs, bar_width, shift=-bar_width)
    quant_discrete_subplot(layer_specs, q_axs, bar_width, shift=0)
    # q_axs.invert_yaxis()

    p_axs.set_xlabel("Network Layers", fontsize=font_size)
    p_axs.set_xticks(ticks=np.arange(len(filtered_layers)), labels=filtered_layers, rotation=90, fontsize=font_size)
    fig.suptitle(title, fontsize=font_size)
    make_legend([p_axs, q_axs], legend_loc)

    return fig


def make_legend(axes, legend_loc):
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc=legend_loc, fontsize=font_size)


def p_discrete_plot(policy, compression_protocol=None, figsize=(16, 3), legend_loc='lower right', title=""):
    layer_specs, filtered_layers = prepare_plots(policy)

    bar_width = 0.4

    fig, p_axs = plt.subplots(figsize=figsize, nrows=1, sharex=True)
    prune_discrete_subplot(layer_specs, compression_protocol, p_axs, bar_width)
    p_axs.set_xticks(ticks=np.arange(len(filtered_layers)), labels=filtered_layers, rotation=90, fontsize=font_size)
    p_axs.set_xlabel("Network Layers", fontsize=font_size)
    make_legend([p_axs], legend_loc)
    fig.suptitle(title, fontsize=font_size)


def q_discrete_plot(policy, figsize=(16, 3), legend_loc='lower right', title=""):
    layer_specs, filtered_layers = prepare_plots(policy)

    bar_width = 0.4

    fig, q_axs = plt.subplots(figsize=figsize, nrows=1, sharex=True)
    quant_discrete_subplot(layer_specs, q_axs, bar_width, shift=-0.5 * bar_width)
    q_axs.set_xticks(ticks=np.arange(len(filtered_layers)), labels=filtered_layers, rotation=90, fontsize=font_size)
    q_axs.set_xlabel("Network Layers", fontsize=font_size)
    make_legend([q_axs], legend_loc)
    fig.suptitle(title, fontsize=font_size)


def prepare_plots(policy):
    layer_keys = list(policy._layers.keys())
    accepted_layers = ["conv", "downsample.0", "fc"]
    filtered_layers = [label for label in layer_keys if any(s in label for s in accepted_layers)]
    layer_specs = {layer_key: spec for layer_key, spec in policy._layers.items() if layer_key in filtered_layers}

    count_dict = {'c': 1, 'l': 1}
    renamed_layers = []
    for layer_key in filtered_layers:
        if "conv" in layer_key or "down" in layer_key:
            renamed_layers.append(f"C{count_dict['c']}")
            count_dict['c'] = count_dict['c'] + 1
        else:
            renamed_layers.append(f"L{count_dict['l']}")
            count_dict['l'] = count_dict['l'] + 1

    return layer_specs, renamed_layers


def prune_discrete_subplot(layer_specs, compression_protocol, p_axs, bar_width, shift=0):
    margin = 0.05
    for idx, (layer_key, layer_dict) in enumerate(layer_specs.items()):
        if "p-lin" in layer_dict or "p-conv" in layer_dict:
            compression_spec = layer_dict["p-conv"] if "p-conv" in layer_dict else layer_dict["p-lin"]
            prune_param = compression_spec.parameter_by_key("sparsity")
            bar = p_axs.bar(idx + shift, prune_param.target_discrete / prune_param.reference, color="tab:blue",
                            width=bar_width, edgecolor="gray", linewidth=0.75, label="P - Sparsity")[0]
            p_axs.text(bar.get_x() + bar.get_width(), bar.get_height() - margin,
                       f"{prune_param.target_discrete} / {prune_param.reference}", rotation=90, color="white",
                       fontweight='bold', fontsize=font_size_bars, va='top', ha="right")
        elif compression_protocol is not None:
            protocol_dict_prune = parse_compression_protocol(compression_protocol)
            if layer_key in protocol_dict_prune:
                protocol_entry = protocol_dict_prune[layer_key]
                original = protocol_entry.before[0]
                after = protocol_entry.result[0]
                bar = p_axs.bar(idx + shift, after / original, color="lightgray",
                                width=bar_width, edgecolor="gray", linewidth=0.75, label="P - Dep. Sparsity")[0]
                p_axs.text(bar.get_x() + bar.get_width(), bar.get_height() - margin,
                           f"{after} / {original}", rotation=90, color="white",
                           fontweight='bold', fontsize=font_size_bars, va='top', ha="right")
            else:
                bar = p_axs.bar(idx + shift, 1, color="lightgray",
                                width=bar_width, edgecolor="gray", linewidth=0.75, label="P - Dep. Sparsity")[0]
                if layer_key == "layer1.0.conv2" or layer_key == "layer1.1.conv2":
                    p_axs.text(bar.get_x() + bar.get_width(), bar.get_height() - margin,
                               f"{64} / {64}", rotation=90, color="white",
                               fontweight='bold', fontsize=font_size_bars, va='top', ha="right")

    p_axs.set_ylabel("Pruning Sparsity", fontsize=font_size)


def parse_compression_protocol(compression_protocol):
    protocol_dict_prune = dict()
    for element in compression_protocol:
        if element.compression_type == "dep-prune" and element.layer_key not in protocol_dict_prune:
            protocol_dict_prune[element.layer_key] = element
    return protocol_dict_prune


def quant_discrete_subplot(layer_specs, q_axs, bar_width, shift=0):
    margin = 0.4
    for idx, layer_dict in enumerate(layer_specs.values()):
        if "q-mixed" in layer_dict:
            compression_spec = layer_dict["q-mixed"]
            activation = compression_spec.parameter_by_key("activation")
            weight = compression_spec.parameter_by_key("weight")
            first_bar = \
                q_axs.bar(idx + shift, activation.target_discrete, color="tab:pink", width=bar_width, edgecolor="gray",
                          hatch="/", linewidth=0.75, label="Q-MIX - activation")[0]
            q_axs.text(first_bar.get_x() + bar_width, first_bar.get_height() - margin,
                       f"a: {activation.target_discrete}",
                       rotation=90, color="white", fontweight='bold', fontsize=font_size_bars, va='top', ha="right")
            second_bar = q_axs.bar(idx + shift + bar_width, weight.target_discrete, color="tab:olive", width=bar_width,
                                   edgecolor="gray", hatch="/", linewidth=0.75, label="Q-MIX - weight")[0]
            q_axs.text(second_bar.get_x() + bar_width, second_bar.get_height() - margin, f"w: {weight.target_discrete}",
                       rotation=90, color="white", fontweight='bold', fontsize=font_size_bars, va='top', ha="right")

        elif "q-int8" in layer_dict:
            bar = \
                q_axs.bar(idx + shift, 8, color="tab:green", width=bar_width, edgecolor="gray", hatch="/",
                          linewidth=0.75,
                          label="Q-INT8")[0]
            q_axs.text(bar.get_x() + bar_width, bar.get_height() - margin, f"INT8", rotation=90, color="white",
                       fontweight='bold',
                       fontsize=font_size_bars, va='top', ha="right")

        elif "q-fp32" in layer_dict:
            bar = q_axs.bar(idx + shift, 32, color="tab:orange", width=bar_width, edgecolor="gray", hatch="/",
                            linewidth=0.75, label="Q-FP32")[0]
            q_axs.text(bar.get_x() + bar_width, bar.get_height() - margin, f"FLOAT32", rotation=90, color="white",
                       fontweight='bold', fontsize=font_size_bars, va='top', ha="right")
    q_axs.set_ylabel("Quantization Bits", fontsize=font_size)
    q_axs.set_ylim(0, 34)

    # q_axs.set_ylim(0, 32)
