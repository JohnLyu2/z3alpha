Running the Sample Script

```bash
./scripts/run_MachSMT_sample.sh
```

Results saved in experiments folder

When setting up MachSMT, use 

```bash
make dev
```

in a virtual environment. 

If does not work (WARNING: The user site-packages directory is disabled), try

```bash
pip install -e .
```

Make sure Machsmt module points to the source code, not in the site-packages directory, so the changes in the code  will take effect without having to reinstall.

Check with:

```bash
python -c "import machsmt; print(machsmt.__file__)"
```