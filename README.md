# stormy
This repository contains scripts for stochastically generating weather time series for use in urban coastal flood risk analysis in Norfolk, VA.

# Cloning repo
## checking out stormy
based on: https://stackoverflow.com/questions/1911109/how-do-i-clone-a-specific-git-branch/7349740#7349740
```
BRANCH=working_branch
REMOTE_REPO=https://github.com/lassiterdc/stormy.git
DIR=stormy
mkdir $DIR
cd $DIR
git init
git remote add -t $BRANCH -f origin $REMOTE_REPO
git checkout $BRANCH
```
## checking out branched RainyDay repo
based on: https://stackoverflow.com/questions/1811730/how-do-i-work-with-a-git-repository-within-another-repository
```
cd stochastic_storm_transposition
REMOTE_REPO=https://github.com/lassiterdc/RainyDay2.git
git init
git submodule add $REMOTE_REPO
```

