# LTL CEGIS

Introductory code for CEGIS for LTL properties applied to Robotics safety and liveness properties.

## Dependencies

- Java JDK ==11 (just JRE will not suffice) (1.8 does not work, 17 does not work)
- Maven >3.0

## Setup Instructions (Linux)

### Setting up Java and Maven without `sudo`

- Download the `jdk-11.0.16_linux-x64_bin.tar.gz` file for getting Java JDK 11 from [here](https://www.oracle.com/java/technologies/javase/jdk11-archive-downloads.html), and extract it in the home directory.
```bash
tar -xvzf "jdk-11.0.16_linux-x64_bin.tar.gz"
mv "jdk-11.0.16/" "$HOME"
```

- Download the `apache-maven-3.8.6-bin.tar.gz` file for getting Maven 3.8.6 from [here](https://maven.apache.org/download.cgi), and extract it in the home directory.
```bash
wget "https://dlcdn.apache.org/maven/maven-3/3.8.6/binaries/apache-maven-3.8.6-bin.tar.gz"
tar -xvzf "apache-maven-3.8.6-bin.tar.gz"
mv "apache-maven-3.8.6/" "$HOME"
```

### Set up LTL2BA

- Get LTL2BA from [here](http://www.lsv.fr/~gastin/ltl2ba/ltl2ba-1.3.tar.gz) and extract it in the home directory.
```bash
wget "http://www.lsv.fr/~gastin/ltl2ba/ltl2ba-1.3.tar.gz"
tar -xvzf "ltl2ba-1.3.tar.gz"
cd ltl2ba-1.3
make
cd ..
mv "ltl2ba-1.3" "$HOME"
```

- Add the following lines at the end of your `~/.bashrc` file, and then run `source ~/.bashrc`:
```bash
JAVA_HOME="$HOME/jdk-11.0.16"
MAVEN_HOME="$HOME/apache-maven-3.8.6"
LTL2BA_HOME="$HOME/ltl2ba-1.3"
PATH="$LTL2BA_HOME:$JAVA_HOME/bin:$MAVEN_HOME/bin:$PATH"

export PATH
export LTL2BA_HOME
export MAVEN_PATH
export JAVA_PATH
export JRE_HOME
```