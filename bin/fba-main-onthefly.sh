#!/bin/bash

# this is necessary for the bash script to have access to the modules system
# using the msdos user at KPNO.
# I could alternatively make the script a --login script, but that seems like
# it's doing more than I might want.
# I haven't understood why one doesn't have access to the modules script when
# coming in from the msdos native csh environment.

if [ -z $NERSC_HOST ] && ! `module list &>/dev/null`; then
    source /etc/profile.d/modules.sh
fi

# this script is a wrapper for a fba_launch call for one main tile
# there is no "svn up" here, no SVN checks
# the code will exit if a TILEID already existing in the official SVN folder exists

TILEID=$1 # e.g. 1000
QA=$2   # y or n

OUTDIR=$FA_HOLDING_PEN
DTCATVER=1.1.1

# setting the proper environment
# https://desi.lbl.gov/trac/wiki/SurveyOps/FiberAssignAtKPNO version #18
if [ -z $NERSC_HOST ]; then
    # at KPNO
    export DESI_PRODUCT_ROOT=/software/datasystems/desiconda/20200924
    export DESI_ROOT=/data/datasystems
    export DESI_TARGET=$DESI_ROOT/target
    export DESI_SURVEYOPS=$DESI_ROOT/survey/ops/surveyops/trunk
    export XDG_CACHE_HOME=$HOME/users/datasystems/xdg_cache_home
    export XDG_CONFIG_HOME=$HOME/users/datasystems/xdg_config_home
    if [ -d $XDG_CACHE_HOME/astropy ]; then
	echo Missing directory $XDG_CACHE_HOME !
    fi
    if [ -d $XDG_CONFIG_HOME/astropy ]; then
	echo Missing directory $XDG_CONFIG_HOME !
    fi
    module use $DESI_PRODUCT_ROOT/modulefiles
    module load desiconda
    module load desimodules/21.5
    module swap desitarget/1.2.2
    module swap fiberassign/5.1.1
    module swap desimeter/0.6.7
    module swap desimodel/master
else
    echo Fiberassign on the fly should only be run for testing purposes at NERSC!
    # source /global/cfs/cdirs/desi/software/desi_environment.sh 21.5
fi

TILESFN=$DESI_SURVEYOPS/ops/tiles-main.ecsv

if [ -z $FIBER_ASSIGN_DIR ]; then
    SVNTILEDIR=$DESI_TARGET/fiberassign/tiles/trunk
else
    SVNTILEDIR=$FIBER_ASSIGN_DIR
fi

# grabbing the RA, DEC, PROGRAM, STATUS, HA for TILEID
LINE=`awk '{if ($1 == '$TILEID') print $0}' $TILESFN`
RA=`echo $LINE | awk '{print $3}'`
DEC=`echo $LINE | awk '{print $4}'`
PROGRAM=`echo $LINE | awk '{print $5}'`
STATUS=`echo $LINE | awk '{print $8}'`
HA=`echo $LINE | awk '{print $10}'`

# small sanity check
if [[ "$STATUS" != "unobs" ]]
then
    echo "TILEID=$TILEID has already been observed; exiting"
    exit
fi

# calling fba_launch
if [[ "$QA" == "y" ]]
then
    CMD="fba_launch --outdir $OUTDIR --tileid $TILEID --tilera $RA --tiledec $DEC --survey main --program $PROGRAM --ha $HA --dtver $DTCATVER --steps qa --log-stdout --doclean y --svntiledir $SVNTILEDIR --worldreadable"
else
    CMD="fba_launch --outdir $OUTDIR --tileid $TILEID --tilera $RA --tiledec $DEC --survey main --program $PROGRAM --ha $HA --dtver $DTCATVER --nosteps qa --doclean n --svntiledir $SVNTILEDIR --worldreadable"
fi
echo $CMD
eval $CMD
