#!/bin/sh

cd "$(dirname "$0")"
PARAMS=
if [ -n "$1" ]
then
	PARAMS=-Ddlchat.shift=$1
fi
MAVEN_OPTS="-Xmx26G" mvn clean compile exec:java -Dexec.mainClass="dlchat.Main" $PARAMS
