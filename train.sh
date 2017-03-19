#!/bin/sh

cd "$(dirname "$0")"
MAVEN_OPTS="-Xmx26G" mvn clean compile exec:java -Dexec.mainClass="dlchat.EncoderDecoderLSTM"
