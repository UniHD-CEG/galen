import inspect

from tvm.relay import expr as _expr, op as _op
from tvm.relay.frontend.pytorch import PyTorchOpConverter

from runtime.compress.compression_policy import CompressionPolicy, LayerCompressionSpecification

BIT_SER_KERNEL_LAYOUT = "HWIO"
BIT_SER_DATA_LAYOUT = "NHWC"
TORCH_KERNEL_LAYOUT = 'OIHW'
TORCH_DATA_LAYOUT = 'NCHW'


class QuantizationOperatorMapper:

    def __init__(self, compression_policy: CompressionPolicy, unipolar=False):
        self._policy = compression_policy.get_clean_policy()
        self._unipolar = False

    def convolution(self, inputs, input_types):
        converter = self._get_original_converter()
        return self._exec_construction_method(inputs, input_types,
                                              int8_map=self._conv_int8,
                                              mixed_map=self._conv_mixed,
                                              original_map=converter.convolution,
                                              converter_ref=converter)

    def linear(self, inputs, input_types):
        converter = self._get_original_converter()
        return self._exec_construction_method(inputs, input_types,
                                              int8_map=self._linear_int8,
                                              mixed_map=self._linear_mixed,
                                              original_map=converter.linear,
                                              converter_ref=converter)

    def _exec_construction_method(self, inputs, input_types, int8_map, mixed_map, original_map, converter_ref):
        weight = inputs[1]
        layer_key = self._get_layer_key_from_weight(weight)

        if self._as_int8(layer_key):
            return int8_map(inputs, input_types, converter_ref)

        if self._as_mixed(layer_key):
            return mixed_map(inputs, input_types, self._as_mixed(layer_key), converter_ref)

        return original_map(inputs, input_types)

    def _get_layer_key_from_weight(self, weight) -> str:
        return weight.astext().split('\n')[1].split('%')[1].split(':')[0][:-7]

    def _get_original_converter(self) -> PyTorchOpConverter:
        stack = inspect.stack()
        caller_self = stack[2][0].f_locals["self"]
        return caller_self

    def _as_mixed(self, layer_key) -> LayerCompressionSpecification | None:
        return self._get_spec(layer_key, "q-mixed")

    def _as_int8(self, layer_key) -> LayerCompressionSpecification | None:
        return self._get_spec(layer_key, "q-int8")

    def _get_spec(self, layer_key, key) -> LayerCompressionSpecification | None:
        if layer_key in self._policy.layers:
            if key in self._policy.layers[layer_key]:
                return self._policy.layers[layer_key][key]
        return None

    def _conv_mixed(self, inputs, input_types, param: LayerCompressionSpecification, converter: PyTorchOpConverter):
        # Does not support transposed convolutions
        data = inputs[0]
        weight = inputs[1]
        bias = inputs[2]
        strides = tuple(inputs[3])
        padding = tuple(inputs[4])

        if isinstance(weight, _expr.Expr):
            inferred_shape = converter.infer_shape(weight)
            weight_shape = []
            for infer in inferred_shape:
                weight_shape.append(infer)
        else:
            msg = "Data type %s could not be parsed in conv op" % (type(weight))
            raise AssertionError(msg)

        out_channels = weight_shape[0]
        in_channels = weight_shape[1]
        kernel_size = weight_shape[2:]
        use_bias = isinstance(bias, _expr.Expr)

        # Conv1d not supported by bitserial use conv2d instead
        is_conv1d = False
        if len(kernel_size) == 1:
            is_conv1d = True
            kernel_size = [1] + kernel_size
            strides = (1,) + strides
            padding = (0,) + padding
            data = _op.expand_dims(data, axis=2)
            weight = _op.expand_dims(weight, axis=2)

        activation_bits = param.parameter_by_key("activation").target_discrete
        weight_bits = param.parameter_by_key("weight").target_discrete

        # flip and cast activation and weight tensors
        data = _op.layout_transform(data, TORCH_DATA_LAYOUT, BIT_SER_DATA_LAYOUT)
        data = _op.cast(data, 'int16')
        weight = _op.layout_transform(weight, TORCH_KERNEL_LAYOUT, BIT_SER_KERNEL_LAYOUT)
        weight = _op.cast(weight, 'uint32')
        # weight_name = param.layer_key + "._conv2d_weight"
        # weight = relay.var(weight_name, relay.TensorType((kernel_size[0], kernel_size[1], in_channels, out_channels),
        #                                                  "uint32"))

        conv_out = _op.nn.bitserial_conv2d(
            data,
            weight,
            strides=strides,
            padding=padding,
            channels=out_channels,
            kernel_size=kernel_size,
            data_layout=BIT_SER_DATA_LAYOUT,
            kernel_layout=BIT_SER_KERNEL_LAYOUT,
            activation_bits=activation_bits,
            weight_bits=weight_bits,
            pack_dtype="uint8",
            out_dtype="int16",
            unipolar=self._unipolar
        )
        conv_out = _op.cast(conv_out, "float32")
        conv_out = _op.layout_transform(conv_out, BIT_SER_DATA_LAYOUT, TORCH_DATA_LAYOUT)
        if use_bias:
            res = _op.nn.bias_add(conv_out, bias)
        else:
            res = conv_out
        if is_conv1d:
            # Because we conducted grouped conv1d convolution through conv2d we must
            # squeeze the output to get the correct result.
            res = _op.squeeze(res, axis=[2])
        return res

    def _linear_mixed(self, inputs, input_types, param: LayerCompressionSpecification, converter: PyTorchOpConverter):
        bias = inputs[2]

        activation_bits = param.parameter_by_key("activation").target_discrete
        weight_bits = param.parameter_by_key("weight").target_discrete
        weight = inputs[1]
        weight_shape = converter.infer_shape_with_prelude(weight)
        data = inputs[0]
        data = _op.cast(data, 'int16')
        weight = _op.cast(weight, 'uint32')

        dense_out = _op.nn.bitserial_dense(data,
                                           weight,
                                           units=weight_shape[0],
                                           data_bits=activation_bits,
                                           weight_bits=weight_bits,
                                           pack_dtype="uint8",
                                           out_dtype="int16",
                                           unipolar=self._unipolar)

        dense_out = _op.cast(dense_out, "float32")

        return self._bias_add(bias, dense_out, input_types, converter)

    def _conv_int8(self, inputs, input_types, converter: PyTorchOpConverter):
        # Use transpose or normal
        use_transpose = True if inputs[6] == 1 else False

        data = inputs[0]
        weight = inputs[1]
        bias = inputs[2]
        strides = tuple(inputs[3])
        padding = tuple(inputs[4])
        dilation = tuple(inputs[5])

        if isinstance(weight, _expr.Expr):
            inferred_shape = converter.infer_shape(weight)
            weight_shape = []
            for infer in inferred_shape:
                weight_shape.append(infer)
        else:
            msg = "Data type %s could not be parsed in conv op" % (type(weight))
            raise AssertionError(msg)

        groups = int(inputs[8])

        if use_transpose:
            channels = weight_shape[1] * groups
            in_channels = weight_shape[0]
        else:
            channels = weight_shape[0]
            in_channels = weight_shape[1]

        # Check if this is depth wise convolution
        # We need to reshape weight so that Relay could recognize this is depth wise
        # weight_shape[1] is always in_channels // groups
        # For depthwise, in_channels == groups, so weight_shape[1] == 1
        # If groups > 1 but weight_shape[1] != 1, this is group convolution
        if groups > 1 and in_channels == 1:
            channel_multiplier = channels // groups
            new_weight_shape = (groups, channel_multiplier) + tuple(weight_shape[2:])
            weight = _op.transform.reshape(weight, new_weight_shape)

        kernel_size = weight_shape[2:]
        use_bias = isinstance(bias, _expr.Expr)

        # We are trying to invoke various relay operations through a single conv_op variable.
        # However the function signatures for some operations have additional attributes so we
        # pass these in along with the standard ones.
        additional_arguments = dict()

        if use_transpose:
            if len(kernel_size) == 3:
                conv_op = _op.nn.conv3d_transpose
            elif len(kernel_size) == 2:
                conv_op = _op.nn.conv2d_transpose
            else:
                conv_op = _op.nn.conv1d_transpose
            output_padding = tuple(inputs[7])
            additional_arguments["output_padding"] = output_padding

        else:
            if len(kernel_size) == 3:
                conv_op = _op.nn.conv3d
            elif len(kernel_size) == 2:
                conv_op = _op.nn.conv2d
            else:
                conv_op = _op.nn.conv1d

        if len(kernel_size) == 3:
            data_layout = "NCDHW"
            kernel_layout = "OIDHW"
        elif len(kernel_size) == 2:
            data_layout = "NCHW"
            kernel_layout = "OIHW"
            if use_transpose:
                # Transposed convolutions have IOHW layout.
                kernel_layout = "IOHW"
        else:
            data_layout = "NCW"
            kernel_layout = "OIW"

        # Conv1d does not currently support grouped convolution so we convert it to conv2d
        is_grouped_conv1d = False
        if groups > 1 and len(kernel_size) == 1 and not use_transpose:
            is_grouped_conv1d = True
            conv_op = _op.nn.conv2d
            kernel_size = [1] + kernel_size
            strides = (1,) + strides
            padding = (0,) + padding
            dilation = (1,) + dilation
            data = _op.expand_dims(data, axis=2)
            weight = _op.expand_dims(weight, axis=2)
            data_layout = "NCHW"
            kernel_layout = "OIHW"

        data = _op.cast(data, "uint8")
        weight = _op.cast(weight, "int8")
        conv_out = conv_op(
            data,
            weight,
            strides=strides,
            padding=padding,
            dilation=dilation,
            groups=groups,
            channels=channels,
            kernel_size=kernel_size,
            data_layout=data_layout,
            kernel_layout=kernel_layout,
            out_dtype="int32",
            **additional_arguments,
        )
        conv_out = _op.cast(conv_out, "float32")
        if use_bias:
            res = _op.nn.bias_add(conv_out, bias)
        else:
            res = conv_out
        if is_grouped_conv1d:
            # Because we conducted grouped conv1d convolution through conv2d we must
            # squeeze the output to get the correct result.
            res = _op.squeeze(res, axis=[2])
        return res

    def _linear_int8(self, inputs, input_types, converter: PyTorchOpConverter):
        bias = inputs[2]
        data = inputs[0]
        weight = inputs[1]
        data = _op.cast(data, "uint8")
        weight = _op.cast(weight, "int8")
        dense_out = _op.nn.dense(data,
                                 weight,
                                 out_dtype="int32")
        dense_out = _op.cast(dense_out, "float32")
        return self._bias_add(bias, dense_out, input_types, converter)

    @staticmethod
    def _bias_add(bias, dense_out, input_types, converter):
        if isinstance(bias, _expr.Expr):
            bias_ndims = len(converter.infer_shape_with_prelude(bias))
            if bias_ndims == 1:
                return _op.nn.bias_add(dense_out, bias, axis=-1)
            dense_dtype = converter.infer_type_with_prelude(dense_out).dtype
            return converter.add([dense_out, bias], [dense_dtype, input_types[2]])
        return dense_out
