from bayesian_network.common.torch_settings import TorchSettings
from bayesian_network.bayesian_network import Node, BayesianNetworkBuilder


class FeatureMap:
    def __init__(
        self,
        node: Node,
        y_range: tuple[int, int],
        x_range: tuple[int, int],
    ):
        self.node = node
        self._y_range = y_range
        self._x_range = x_range

    def _in_range(self, range, point) -> bool:
        return (range[0] <= point) and (point < range[1])

    def _in_y_range(self, y):
        return self._in_range(self._y_range, y)

    def _in_x_range(self, x):
        return self._in_range(self._x_range, x)

    def contains(self, y: int, x: int):
        return self._in_y_range(y) and self._in_x_range(x)


class Network1LBuilder:
    def __init__(self, torch_settings: TorchSettings, height: int, width: int):
        self._torch_settings = torch_settings
        self._height = height
        self._width = width

    def build(self, numQ: int, numF: int, kernel_size: int, stride: int):
        builder = BayesianNetworkBuilder()

        # Q
        Q = Node.random((numQ), self._torch_settings, name="Q")
        builder.add_node(Q)

        # Fs
        feature_maps = []
        Fs = []
        for ify in range(0, self._height - kernel_size + 1, stride):
            for ifx in range(0, self._width - kernel_size + 1, stride):
                F = Node.random((numQ, numF), self._torch_settings, name=f"F_{ify}x{ifx}")
                Fs.append(F)
                builder.add_node(F, parents=Q)

                feature_map = FeatureMap(
                    F,
                    y_range=(ify, ify + kernel_size),
                    x_range=(ifx, ifx + kernel_size),
                )

                feature_maps.append(feature_map)

        # Ys
        Ys = []
        for iy in range(self._height):
            for ix in range(self._width):
                parents = [
                    feature_map.node for feature_map in feature_maps if feature_map.contains(iy, ix)
                ]

                dims = ([numF] * len(parents)) + [2]
                Y = Node.random(dims, self._torch_settings, name=f"y_{iy}x{ix}")

                Ys.append(Y)
                builder.add_node(Y, parents=parents)

        return builder.build(), Q, Fs, Ys


class Network3LBuilder:
    def __init__(self, torch_settings: TorchSettings):
        self._torch_settings = torch_settings
        self._height = 22
        self._width = 22

    def build(self, numF0: int, numF1: int, numF2: int):
        builder = BayesianNetworkBuilder()

        # F0
        F0 = Node.random((numF0), self._torch_settings, name="F0")
        builder.add_node(F0)

        # F1s
        kernel_size = 10
        F1_feature_maps = []
        F1s = []
        for iy in range(0, self._height - kernel_size + 1, 4):
            for ix in range(0, self._width - kernel_size + 1, 4):
                F1 = Node.random((numF0, numF1), self._torch_settings, name=f"F1_{iy}x{ix}")
                F1s.append(F1)
                builder.add_node(F1, parents=F0)

                feature_map = FeatureMap(
                    F1,
                    y_range=(iy, iy + kernel_size),
                    x_range=(ix, ix + kernel_size),
                )

                F1_feature_maps.append(feature_map)

        # F2s
        kernel_size = 4
        F2_feature_maps = []
        F2s = []
        for iy in range(0, self._height - kernel_size + 1, 2):
            for ix in range(0, self._width - kernel_size + 1, 2):
                parents = [
                    feature_map.node
                    for feature_map in F1_feature_maps
                    if feature_map.contains(iy, ix)
                ]
                dims = ([numF1] * len(parents)) + [numF2]

                F2 = Node.random(dims, self._torch_settings, name=f"F2_{iy}x{ix}")
                F2s.append(F2)
                builder.add_node(F2, parents=parents)

                feature_map = FeatureMap(
                    F2,
                    y_range=(iy, iy + kernel_size),
                    x_range=(ix, ix + kernel_size),
                )

                F2_feature_maps.append(feature_map)

        # Ys
        Ys = []
        for iy in range(self._height):
            for ix in range(self._width):
                parents = [
                    feature_map.node
                    for feature_map in F2_feature_maps
                    if feature_map.contains(iy, ix)
                ]

                dims = ([numF2] * len(parents)) + [2]
                Y = Node.random(dims, self._torch_settings, name=f"y_{iy}x{ix}")

                Ys.append(Y)
                builder.add_node(Y, parents=parents)

        return builder.build(), F0, F1s, F2s, Ys
