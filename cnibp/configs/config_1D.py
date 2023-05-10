import yaml

'''https://www.inflearn.com/questions/16184/yaml%ED%8C%8C%EC%9D%BC-%EC%9D%B4%EB%9E%80-%EB%AC%B4%EC%97%87%EC%9D%B8%EA%B0%80%EC%9A%94'''


class CFG:
    def __init__(self, yaml_path):
        if isinstance(yaml_path, str):
            with open(yaml_path) as f:
                yaml_data = yaml.safe_load(f)
        else:
            yaml_data = yaml_path
        for key, value in yaml_data.items():
            if isinstance(value, dict):
                setattr(self, key, CFG(value))
            else:
                setattr(self, key, value)

    def get_config(file_path):
        cfg = CFG(file_path)
        return cfg

    if __name__ == '__main__':
        cfg = get_config('./configs/config_1D.yaml')
        print(cfg)
