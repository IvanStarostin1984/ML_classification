from src import dataset_summary


def test_dataset_summary_is_callable() -> None:
    assert callable(dataset_summary)
