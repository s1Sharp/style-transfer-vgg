#!/bin/sh

## Usage:
##   . ./export-env.sh ; $COMMAND
##   . ./export-env.sh ; echo ${TG_NOTIFY_BOT_TOKEN}
## 
##   . ./export-env.sh ; echo ${TG_NOTIFY_BOT_TOKEN}


unamestr=$(uname)
envfile="./config/.env"

if [ -z "$1" ]
  then
    echo "No argument supplied, apply default $envfile name"
  elif [ ! -f "$1" ]; then
      echo "File not found!"
  else
    echo "Set custom env filename $1"
    envfile="$1"
fi

if [ "$unamestr" = 'Linux' ]; then
  export $(grep -v '^#' "$envfile" | xargs -d '\n')
  echo "Set linux $envfile vars ok"
elif [ "$unamestr" = 'FreeBSD' ] || [ "$unamestr" = 'Darwin' ]; then
  export $(grep -v '^#' "$envfile" | xargs -0)
  echo "Set $envfile vars ok"
fi