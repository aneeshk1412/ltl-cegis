# LTL CEGIS

Introductory code for CEGIS for LTL properties applied to Robotics safety and liveness properties.

## Dependencies

- Java JDK >=11 (just JRE will not suffice)
- Maven >3.0

## Setup Instructions (Linux)

### Setting up Java and Maven without `sudo`

- Download the `jdk-17_linux-x64_bin.tar.gz` file for getting [Java JDK 17](https://download.oracle.com/java/17/latest/jdk-17_linux-x64_bin.tar.gz), and extract it in the home directory.
```bash
wget "https://download.oracle.com/java/17/latest/jdk-17_linux-x64_bin.tar.gz"
tar -xvzf "jdk-17_linux-x64_bin.tar.gz"
mv "jdk-17.0.4.1/" "$HOME"
```

- Download the `apache-maven-3.8.6-bin.tar.gz` file for getting [Maven 3.8.6](), and extract it in the home directory.
```bash
wget "https://dlcdn.apache.org/maven/maven-3/3.8.6/binaries/apache-maven-3.8.6-bin.tar.gz"
tar -xvzf "apache-maven-3.8.6-bin.tar.gz"
mv "apache-maven-3.8.6/" "$HOME"
```

- Add the following lines at the end of your `~/.bashrc` file, and then run `source ~/.bashrc`:
```bash
JAVA_HOME="$HOME/jdk-17.0.4.1"
MAVEN_HOME="$HOME/apache-maven-3.8.6"

PATH="$JAVA_HOME/bin:$MAVEN_HOME/bin:$PATH"

export PATH
export MAVEN_PATH
export JAVA_PATH
export JRE_HOME
```
