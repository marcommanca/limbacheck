import json
import os

class ConfigUtils:

    def __init__(self):
        if os.path.exists('LMSpellConfig.json'):
            with open('LMSpellConfig.json', 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {}
   
    def get(self, key, default=None):
        keys = key.split('.')
        value = self.config
        try:
            for k in keys:
                value = value[k]
            return value
        except KeyError:
            return default


    @staticmethod
    def get_dataset_columns():
        config = ConfigUtils()
        return [
            config.get('dataset.inputs', 'text'),
            config.get('dataset.targets', 'expected')
        ]
    
    @staticmethod
    def get_results_columns():
        config = ConfigUtils()
        return [
            config.get('results.original', 'Original'),
            config.get('results.predicted', 'Corrected'),
            config.get('results.expected', 'Expected')
        ]