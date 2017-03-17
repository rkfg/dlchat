#!/bin/sh
MAVEN_OPTS="-Xmx26G -XX:+UseG1GC" mvn clean compile exec:java -Dexec.mainClass="dlchat.Main"
