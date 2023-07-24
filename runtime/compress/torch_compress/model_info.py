import torchinfo
from torchinfo.layer_info import LayerInfo

from runtime.model.torch_model import TorchModelReference


class ModelInfo:

    def __init__(self, model: TorchModelReference):
        self.layer_list = torchinfo.summary(model._reference_model, model.batch_input_shape, verbose=0).summary_list

    def get_info_for_layer(self, layer_key) -> LayerInfo:
        return self._get_info(layer_key, self.layer_list, layer_key, self.layer_list[0])

    def _get_info(self, layer_key: str, layer_list: list[LayerInfo], full_key, parent_info) -> LayerInfo:
        key_elements = layer_key.split(".")

        if len(key_elements) > 1:
            if parent_info.parent_info:
                parents = {info.var_name: info for info in layer_list if not info.is_leaf_layer and info.parent_info.var_name == parent_info.var_name}
            else:
                parents = {info.var_name: info for info in layer_list if not info.is_leaf_layer}

            if key_elements[0] in parents:
                current_info = parents[key_elements[0]]
                return self._get_info(".".join(key_elements[1:]), current_info.children, full_key, current_info)

        leafs = {info.var_name: info for info in layer_list if
                 info.is_leaf_layer and info.parent_info.var_name == parent_info.var_name}
        if key_elements[0] in leafs:
            return leafs[key_elements[0]]

        raise Exception(f"Could not resolve layer info for {full_key} - step failed for part {'.'.join(key_elements)}")