import yaml

class CFG:
    def __init__(self, yaml_path):
        if isinstance(yaml_path, str):
            with open(yaml_path, "r") as f:
                yaml_data = yaml.safe_load(f)
        else:
            yaml_data = yaml_path

        for key, value in yaml_data.items():
            if isinstance(value, dict):
                setattr(self, key, CFG(value))
            else:
                setattr(self, key, value)

def get_config(file):
    cfg = CFG(file)
    return cfg

if __name__ == '__main__':
    cfg = get_config("./configs/base_config.yaml")
    print(cfg)