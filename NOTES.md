# Migration notes

Current commit: `fb5350f`.

At this stage the repository holds the original `ai_arisha.py` script plus the
initial project skeleton under `src/`, `scripts/`, `tests/` and `notebooks`.
A basic GitHub Actions workflow performs linting and tests on every push.

The notebook script still contains many Colab-specific commands such as
`files.upload()` and shell calls (`!pip install`, `!kaggle datasets download`).
It also mixes data cleaning, feature engineering and model training in one file.

The next step is to break this large script into smaller modules as outlined in
`TODO.md` and introduce tests plus GitHub Actions.

2025-04-30: Added environment.yml, requirements.txt, Dockerfile, Makefile, .gitignore and LICENSE to start project skeleton.
2025-06-08: Created module, script and notebook directories and introduced GitHub Actions CI workflow (`fb5350f`).
