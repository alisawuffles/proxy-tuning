from pathlib import Path
import yaml

# Config
CONFIG_FILE = Path('config.yml')
with open(CONFIG_FILE) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
MY_OPENAI_API_KEY = config['my-openai-key']
AI2_OPENAI_API_KEY = config['ai2-openai-key']
