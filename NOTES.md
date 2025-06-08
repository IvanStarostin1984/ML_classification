# Migration notes

Current commit: `dbd5184`.

At this point the repo consists only of the original Colab export `ai_arisha.py`,
a project `README.md`, `AGENTS.md` and a data licence notice under `data/`.
None of the planned `src/` modules or CI files exist yet.

The notebook script still contains many Colab-specific commands such as
`files.upload()` and shell calls (`!pip install`, `!kaggle datasets download`).
It also mixes data cleaning, feature engineering and model training in one file.

The next step is to break this large script into smaller modules as outlined in
`TODO.md` and introduce tests plus GitHub Actions.

2025-04-30: Added environment.yml, requirements.txt, Dockerfile, Makefile, .gitignore and LICENSE to start project skeleton.
2025-06-08: Added smoke test importing src and scripts skeleton modules.
