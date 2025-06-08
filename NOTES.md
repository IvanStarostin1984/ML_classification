# Migration notes

Current commit: `8520315`.

At this point the repo consists only of the original Colab export `ai_arisha.py`,
a project `README.md`, `AGENTS.md` and a data licence notice under `data/`.
None of the planned `src/` modules or CI files exist yet.

The notebook script still contains many Colab-specific commands such as
`files.upload()` and shell calls (`!pip install`, `!kaggle datasets download`).
It also mixes data cleaning, feature engineering and model training in one file.

The next step is to break this large script into smaller modules as outlined in
`TODO.md` and introduce tests plus GitHub Actions.

2025-04-30: Added environment.yml, requirements.txt, Dockerfile, Makefile, .gitignore and LICENSE to start project skeleton.
2025-06-08: Set up CI workflow and created src/, scripts/ and tests skeletons with a smoke test.
