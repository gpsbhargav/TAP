#!/bin/bash
sed -e 's/ ##//g' -e 's/\([0-9]\) \, \([0-9]\)/\1\,\2/g' -e "s/ ' /'/g" -e "s/\([^0-9]\) - \([^0-9]\)/\1-\2/g" -e 's/ \. /\./g' predictions.json > pred.json
