import os
import torch
import torch.nn as nn


def get_model_device(model):
    return next(model.parameters()).device


class ModelRegistrar(nn.Module):
    def __init__(self, model_dir, device):
        super(ModelRegistrar, self).__init__()
        self.model_dict = nn.ModuleDict()
        self.model_dir = model_dir
        self.device = device
        self.param_dict = nn.ParameterDict()

    def forward(self):
        raise NotImplementedError(
            "Although ModelRegistrar is a nn.Module, it is only to store parameters."
        )

    def get_model(self, name, model_if_absent=None):
        # 4 cases: name in self.model_dict and model_if_absent is None         (OK)
        #          name in self.model_dict and model_if_absent is not None     (OK)
        #          name not in self.model_dict and model_if_absent is not None (OK)
        #          name not in self.model_dict and model_if_absent is None     (NOT OK)

        if name in self.model_dict:
            return self.model_dict[name]

        elif model_if_absent is not None:
            self.model_dict[name] = model_if_absent.to(self.device)
            return self.model_dict[name]

        else:
            raise ValueError(f"{name} was never initialized in this Registrar!")

    def get_parameter(self, name, param_if_absent):
        if name in self.param_dict:
            return self.param_dict[name]

        elif param_if_absent is not None:
            self.register_parameter(name=name, param=param_if_absent.to(self.device))
            self.param_dict[name] = getattr(self, name)
            return self.param_dict[name]

        else:
            raise ValueError(f"{name} was never initialized in this Registrar!")

    def get_name_match(self, name):
        ret_model_list = nn.ModuleList()
        for key in self.model_dict.keys():
            if name in key:
                ret_model_list.append(self.model_dict[key])
        return ret_model_list

    def get_all_but_name_match(self, name):
        ret_model_list = nn.ModuleList()
        for key in self.model_dict.keys():
            if name not in key:
                ret_model_list.append(self.model_dict[key])
        return ret_model_list

    def print_model_names(self):
        print(self.model_dict.keys())

    def save_models(self, curr_iter):
        # Create the model directiory if it's not present.
        save_path = os.path.join(self.model_dir, "model_registrar-%d.pt" % curr_iter)

        torch.save({"models": self.model_dict, "params": self.param_dict}, save_path)

    def load_models(self, iter_num):
        self.model_dict.clear()
        self.param_dict.clear()

        save_path = os.path.join(self.model_dir, "model_registrar-%d.pt" % iter_num)

        print("")
        print("Loading from " + save_path)
        saved_module = torch.load(save_path, map_location=self.device)
        self.model_dict = saved_module["models"]
        self.param_dict = saved_module["params"]
        print("Loaded!")
        print("")
