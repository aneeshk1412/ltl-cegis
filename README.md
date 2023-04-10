## Installation

Probably create a virtualenv for python related stuff

### Other Python requirements

```sh
pip install -r requirements.txt
```

### Custom Minigrid Installation

```sh
git clone -b base https://github.com/aneeshk1412/Minigrid.git
cd Minigrid
python setup.py build
python setup.py install
```

### Installation of PRISM

- Simply get the PRISM installation for your system from [here](https://www.prismmodelchecker.org/download.php).
- Make sure that `prism` executable can be run from anywhere (add it to your `bin/` and `$PATH` variable)

## Running

Example of running the learning procedure:

```sh
python local_expansion_minigrid.py --env-name MiniGrid-MultiKey-K1-Ordered-16x16-v0 --simulator-seed 100 --learner-seed 100 --show-if-unsat
```
