import json

import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

from models import get_model

#

with open('params.json') as f:
    jsonObject = json.load(f)
    __PREPROCESSING__ = jsonObject.get("__PREPROCESSING__")
    __TIME__ = jsonObject.get("__TIME__")
    __MODEL_SUMMARY__ = jsonObject.get("__MODEL_SUMMARY__")
    options = jsonObject.get("options")
    params = jsonObject.get("params")
    hyper_params = jsonObject.get("hyper_params")
    model_params = jsonObject.get("model_params")
#

convert2version5 = True
if convert2version5:
    from torch.jit.mobile import (
        _backport_for_mobile,
        _get_model_bytecode_version,
    )

    MODEL_INPUT_FILE = params["model_root_path"] + model_params["name"] + params["dataset_name"] + "newtrialWH"
    MODEL_OUTPUT_FILE = params["model_root_path"] + model_params["name"] + params["dataset_name"] + "newtrialWH" + "V5"

    print("model version", _get_model_bytecode_version(f_input=MODEL_INPUT_FILE))

    _backport_for_mobile(f_input=MODEL_INPUT_FILE, f_output=MODEL_OUTPUT_FILE, to_version=5)

    print("new model version", _get_model_bytecode_version(MODEL_OUTPUT_FILE))

"""
TEST FOR LOAD
"""
model = get_model(model_params["name"])
model_dict = torch.load(params["model_root_path"] + model_params["name"] + params["dataset_name"] + "newtrialWH" + "V5")
model.load_state_dict(model_dict)

example = torch.rand(200, 3, 32, 128, 128)
traced_script_module = torch.jit.trace(model, example)
traced_script_module_optimized = optimize_for_mobile(traced_script_module)
traced_script_module_optimized._save_for_lite_interpreter(
    params["model_root_path"] + model_params["name"] + params["dataset_name"] + "mobile")
