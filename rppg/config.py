import yaml

def get_config(file):
    with open(file) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg

if __name__ == '__main__':
    cfg = get_config("./configs/PRE_CONT_UBFC.yaml")
    print(cfg)