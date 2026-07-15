def experiment_id(config):
    value=f"{config['dataset']}-{config['model']}-{config['attention']}"
    if config['attention'] != 'none': value+=f"-lambda{config['lambda']:g}"
    return f"{value}-seed{config['seed']}"
