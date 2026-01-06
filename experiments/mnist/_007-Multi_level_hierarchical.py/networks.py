from bayesian_network.common.torch_settings import TorchSettings
from bayesian_network.bayesian_network import Node, BayesianNetworkBuilder


class Network1LBuilder:
    def __init__(self, torch_settings: TorchSettings, height: int, width: int):
        self._torch_settings = torch_settings
        self._height = height
        self._width = width

    def build(self, dim_F0):
        builder = BayesianNetworkBuilder()

        F0 = Node.random((dim_F0), torch_settings=self._torch_settings, name="F0")
        builder.add_node(F0)

        Ys = []

        for iy in range(self._height):
            for ix in range(self._width):
                Y = Node.random(
                    (dim_F0, 2),
                    torch_settings=self._torch_settings,
                    name=f"Y_{iy}x{ix}",
                )

                Ys.append(Y)

        builder.add_nodes(Ys, parents=F0)

        # Create network
        return builder.build(), F0, Ys


class Network2LBuilder:
    def __init__(
        self,
        torch_settings: TorchSettings,
        height: int,
        width: int,
    ):
        self._torch_settings = torch_settings
        self._height = height
        self._width = width
        self._patch_height = 4
        self._patch_width = 4

    def build(self, dim_F0, dim_F1):
        builder = BayesianNetworkBuilder()

        F0 = Node.random((dim_F0), torch_settings=self._torch_settings, name="F0")
        builder.add_node(F0)

        Ys_dict = {}
        F1s = []

        for if1y in range(self._height // self._patch_height):
            for if1x in range(self._width // self._patch_width):
                F1 = Node.random(
                    (dim_F0, dim_F1),
                    torch_settings=self._torch_settings,
                    name=f"F1_{if1y}x{if1x}",
                )
                builder.add_node(F1, parents=F0)
                F1s.append(F1)

                for iy in range(self._patch_height):
                    for ix in range(self._patch_width):
                        index_y = if1y * self._patch_height + iy
                        index_x = if1x * self._patch_width + ix

                        Y = Node.random(
                            (dim_F1, 2),
                            torch_settings=self._torch_settings,
                            name=f"Y_{index_y}x{index_x}",
                        )
                        builder.add_node(Y, parents=F1)
                        Ys_dict[(index_y, index_x)] = Y

        Ys = [Ys_dict[Y] for Y in sorted(Ys_dict)]

        # Create network
        return builder.build(), F0, F1s, Ys


class Network3LBuilder:
    def __init__(
        self,
        torch_settings: TorchSettings,
        height: int,
        width: int,
    ):
        self._torch_settings = torch_settings
        self._height = height
        self._width = width

    def build(self, dim_F0, dim_F1, dim_F2):
        builder = BayesianNetworkBuilder()

        F0 = Node.random((dim_F0), torch_settings=self._torch_settings, name="F0")
        builder.add_node(F0)

        F1s_dict = {}
        F2s = []
        Ys_dict = {}

        for if1y in range(3):
            for if1x in range(3):
                F1 = Node.random(
                    (dim_F0, dim_F1), torch_settings=self._torch_settings, name=f"F1_{if1y}x{if1x}"
                )
                builder.add_node(F1, parents=F0)
                F1s_dict[(if1y, if1x)] = F1

        for if2y in range(7):
            if if2y >= 5:
                if1y = 2
            elif if2y >= 2:
                if1y = 1
            else:
                if1y = 0

            for if2x in range(7):
                if if2x >= 5:
                    if1x = 2
                elif if2x >= 2:
                    if1x = 1
                else:
                    if1x = 0

                F2 = Node.random(
                    (dim_F1, dim_F2),
                    torch_settings=self._torch_settings,
                    name=f"F2_{if2y}x{if2x}",
                )
                builder.add_node(F2, parents=F1s_dict[(if1y, if1x)])
                F2s.append(F2)

                for iy in range(4):
                    for ix in range(4):
                        index_y = if2y * 4 + iy
                        index_x = if2x * 4 + ix

                        Y = Node.random(
                            (dim_F2, 2),
                            torch_settings=self._torch_settings,
                            name=f"Y_{index_y}x{index_x}",
                        )
                        builder.add_node(Y, parents=F2)
                        Ys_dict[(index_y, index_x)] = Y

        Ys = [Ys_dict[Y] for Y in sorted(Ys_dict)]

        # Create network
        return builder.build(), F0, list(F1s_dict.values()), F2s, Ys


class Network4LBuilder:
    def __init__(
        self,
        torch_settings: TorchSettings,
        height: int,
        width: int,
    ):
        self._torch_settings = torch_settings
        self._height = height
        self._width = width

    def build(self, dim_F0, dim_F1, dim_F2, dim_F3):
        builder = BayesianNetworkBuilder()

        F0 = Node.random((dim_F0), torch_settings=self._torch_settings, name="F0")
        builder.add_node(F0)

        F1s_dict = {}
        F2s = []
        F3s = []
        Ys_dict = {}

        for if1y in range(3):
            for if1x in range(3):
                F1 = Node.random(
                    (dim_F0, dim_F1), torch_settings=self._torch_settings, name=f"F1_{if1y}x{if1x}"
                )
                builder.add_node(F1, parents=F0)
                F1s_dict[(if1y, if1x)] = F1

        for if2y in range(7):
            if if2y >= 5:
                if1y = 2
            elif if2y >= 2:
                if1y = 1
            else:
                if1y = 0

            for if2x in range(7):
                if if2x >= 5:
                    if1x = 2
                elif if2x >= 2:
                    if1x = 1
                else:
                    if1x = 0

                F2 = Node.random(
                    (dim_F1, dim_F2),
                    torch_settings=self._torch_settings,
                    name=f"F2_{if2y}x{if2x}",
                )
                builder.add_node(F2, parents=F1s_dict[(if1y, if1x)])
                F2s.append(F2)

                for if3y in range(2):
                    for if3x in range(2):
                        F3 = Node.random(
                            (dim_F2, dim_F3),
                            torch_settings=self._torch_settings,
                            name=f"F3_{if2y * 2 + if3y}x{if2x * 2 + if3x}",
                        )
                        builder.add_node(F3, parents=F2)
                        F3s.append(F3)

                        for iy in range(2):
                            for ix in range(2):
                                index_y = if2y * 4 + if3y * 2 + iy
                                index_x = if2x * 4 + if3x * 2 + ix

                                Y = Node.random(
                                    (dim_F3, 2),
                                    torch_settings=self._torch_settings,
                                    name=f"Y_{index_y}x{index_x}",
                                )
                                builder.add_node(Y, parents=F3)
                                Ys_dict[(index_y, index_x)] = Y

        Ys = [Ys_dict[Y] for Y in sorted(Ys_dict)]

        # Create network
        return builder.build(), F0, list(F1s_dict.values()), F2s, F3s, Ys
