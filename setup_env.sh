#!/bin/bash

# 1. Update and install Java 17
echo "Installing Java 17..."
sudo apt update
sudo apt install openjdk-17-jdk -y

# 2. Set Java 17 as default
echo "Setting Java 17 as default..."
sudo update-alternatives --set java /usr/lib/jvm/java-17-openjdk-amd64/bin/java

# 3. Export variables for the current session
echo "Setting JAVA_HOME..."
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH

echo "Environment setup complete. Java version:"
java -version