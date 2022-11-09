# Counter-example Guided Inductive Synthesis for Temporal Specifications of Robot Policies

Introductory code for CEGIS for LTL properties applied to Robotics safety and liveness properties.

- Make a virtual environment `virtualenv .venv` inside the LTL-CEGIS repository 
    - (Try to make it with python version 3.10)
- Download these requirements inside the `.venv` folder
    - Java 1.8.0_341 : https://www.oracle.com/java/technologies/javase/javase8u211-later-archive-downloads.html
    - ltl2ba 1.3 : http://www.lsv.fr/~gastin/ltl2ba/ltl2ba-1.3.tar.gz
    - apache maven 3.8.6 : https://dlcdn.apache.org/maven/maven-3/3.8.6/binaries/apache-maven-3.8.6-bin.tar.gz
    - old Ultimate : https://ultimate.informatik.uni-freiburg.de/downloads/ltlautomizer/LTLAutomizer_Linux64_r13621.zip
- Run these 5 commands to move these folders to `.venv` folder
```bash
tar -xvzf apache-maven-3.8.6-bin.tar.gz --directory ltl-cegis/.venv/
tar -xvzf jdk-8u333-linux-x64.tar.gz --directory ltl-cegis/.venv/
tar -xvzf ltl2ba-1.3.tar.gz --directory ltl-cegis/.venv/
unzip LTLAutomizer_Linux64_r13621.zip -d ltl-cegis/.venv/
```
- Change directory to `ltl-cegis/.venv/ltl2ba-1.3` and run `make`
- Change directory `ltl-cegis/.venv/x86_64/` and run `chmod ugoa+x Ultimate`
- Open file `ltl-cegis/.venv/bin/activate` and write these
```bash
# JAVA_HOME="$VIRTUAL_ENV/jdk-11.0.16.1"  ## new Ultimate
JAVA_HOME="$VIRTUAL_ENV/jdk1.8.0_341" ## old Ultimate
MAVEN_HOME="$VIRTUAL_ENV/apache-maven-3.8.6"
LTL2BA_HOME="$VIRTUAL_ENV/ltl2ba-1.3"
# ULTIMATE_HOME="$VIRTUAL_ENV/UAutomizer-linux/" ## new Ultimate
ULTIMATE_HOME="$VIRTUAL_ENV/x86_64/" ## old Ultimate
PATH="$ULTIMATE_HOME:$LTL2BA_HOME:$JAVA_HOME/bin:$MAVEN_HOME/bin:$PATH"

export PATH
export LTL2BA_HOME
export ULTIMATE_HOME
export MAVEN_PATH
export JAVA_PATH
export JRE_HOME
```
- Run activate venv again `source ltl-cegis/.venv/bin/activate`
- Now change directory to `ltl-cegis/src` and run `python test.py`
    - This should run Algorithm 1 (for each enumerated program, check if it verifies the spec, if yes, then stop, else continue)

### Types of Ultimate Results
- RESULT: Ultimate could not prove your program: Toolchain returned no result.
    - --- Results ---
- RESULT: Ultimate proved your program to be correct!
    - All specifications hold
- RESULT: Ultimate proved your program to be incorrect!
    - LTLInfiniteCounterExampleResult

### prog_test.c results
#### 1D Program, do not hit wall, patrolling

Safety SAT : 40s (correct program)
Safety UNSAT : 5s (does not check wall, oscillate)

Liveness SAT : 4min (correct program)
Liveness UNSAT : 10s (does not check wall, oscillate)