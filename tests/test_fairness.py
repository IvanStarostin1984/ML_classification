from src.fairness import four_fifths_ratio


def test_equal_tpr_ratio_is_one():
  y_true = [1, 0, 1, 0]
  y_pred = [1, 0, 1, 0]
  group = ['A', 'A', 'B', 'B']
  assert four_fifths_ratio(y_true, y_pred, group) == 1.0


def test_single_group_returns_one():
  y_true = [1, 0]
  y_pred = [1, 1]
  group = ['A', 'A']
  assert four_fifths_ratio(y_true, y_pred, group) == 1.0


def test_imbalanced_groups_below_threshold():
  y_true = [1, 1, 1, 1]
  y_pred = [1, 1, 0, 0]
  group = ['A', 'A', 'B', 'B']
  ratio = four_fifths_ratio(y_true, y_pred, group)
  assert 0 <= ratio < 0.8
