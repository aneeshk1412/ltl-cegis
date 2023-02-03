# Instructions

```bash
git clone --recurse-submodules ltl-cegis-repo-name
```

```bash
cd ltl-cegis/Minigrid
git checkout master
python3 setup.py build
python3 setup.py install
```

```
cd ltl-cegis/
python cegis_dtree_minigrid.py
```

## TODOs

- Refactor the DSL
  - has_key
  - door_in_front
  - door_is_open
  - Change bfs to only see stuff in direction
- Rename headers for DoorKey