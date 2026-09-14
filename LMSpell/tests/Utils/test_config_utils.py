from unittest.mock import mock_open, patch
from Utils.ConfigUtils import ConfigUtils
import json

def test_get_configs():

    mock_config_data = json.dumps({
        "dataset": {
            "seq2seq": {
                "inputs": "text",
                "targets": "expected"
            }
        }
    })

    with patch("builtins.open", mock_open(read_data=mock_config_data)), patch("os.path.exists", return_value=True):

        config = ConfigUtils()
        assert config.get('dataset.seq2seq.inputs') == 'text'
        assert config.get('dataset.seq2seq.targets') == 'expected'
        assert config.get('dataset.seq2seq.non_existing_key') is None
        assert config.get('non_existing_key', 'default_value') == 'default_value'

def test_get_bool_configs():
    mock_config_data = json.dumps({
        "fix_zwj": {
            "mbart": True,
            "gemma": False
        }
    })
    
    with patch("builtins.open", mock_open(read_data=mock_config_data)), patch("os.path.exists", return_value=True):
        config = ConfigUtils()
        assert config.get('fix_zwj.mbart') is True
        assert config.get('fix_zwj.gemma') is False