from config import registry


def test_loaders():
    assert not registry.load_feature_inventory().empty
    assert not registry.load_join_keys().empty
    assert not registry.load_data_sources().empty
    assert not registry.load_pipelines().empty
    assert not registry.load_leakage_audit().empty
    assert not registry.load_modelling_tasks().empty
