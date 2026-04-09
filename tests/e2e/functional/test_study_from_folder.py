from pathlib import Path

from gems.simulation import TimeBlock
from gems.study.folder import load_study, run_study


def test_load_study():

    study_dir = Path(__file__).parent / 'studies' / '7_4'

    network, database = load_study(study_dir)
    assert len(network.components) == 12
    assert len(network.connections) == 11
    assert len(database._data) == 76


def test_run_study():

    study_dir = Path(__file__).parent / 'studies' / '7_4'

    problem = run_study(study_dir, 1, TimeBlock(0, [0]))
    assert problem.status == 'ok'
    assert problem.objective_value == 100210.
    pass