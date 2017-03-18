#!/bin/sh
MAVEN_OPTS="-Xmx26G" mvn clean compile exec:java -Dexec.mainClass="dlchat.Main"
