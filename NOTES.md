# Migration notes

Current commit: `0d11a70`.

At this point the repo consists only of the original Colab export `ai_arisha.py`,
a project `README.md`, `AGENTS.md` and a data licence notice under `data/`.
None of the planned `src/` modules or CI files exist yet.

The notebook script still contains many Colab-specific commands such as
`files.upload()` and shell calls (`!pip install`, `!kaggle datasets download`).
It also mixes data cleaning, feature engineering and model training in one file.

The next step is to break this large script into smaller modules as outlined in
`TODO.md` and introduce tests plus GitHub Actions.

Initial skeleton modules under `src/` were created along with a smoke test to
ensure they can all be imported.
