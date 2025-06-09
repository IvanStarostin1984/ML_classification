def _dummy_df(n: int = 10):
    import pandas as pd
    return pd.DataFrame({'x': range(n), 'target': [0, 1] * (n // 2)})


def test_split_lengths() -> None:
    from src.split import stratified_split
    df = _dummy_df()
    train, val, test = stratified_split(
        df, 'target', test_size=0.2, val_size=0.2, random_state=0
    )
    assert len(train) + len(val) + len(test) == len(df)


def test_class_proportions_and_indices() -> None:
    from src.split import stratified_split
    df = _dummy_df(20)
    base_prop = df['target'].mean()
    train, val, test = stratified_split(df, 'target', random_state=1)
    for part in (train, val, test):
        assert part['target'].mean() == base_prop
        assert part.index.tolist() == list(range(len(part)))
