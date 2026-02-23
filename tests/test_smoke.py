# To check project installs correctly and the config loader works

from newsclf.utils.config import load_config


def test_load_config_smoke():
    cfg = load_config("configs/base.yaml")
    assert cfg["project"]["name"] == "news-nlp-classifier"
    assert "preprocessing" in cfg