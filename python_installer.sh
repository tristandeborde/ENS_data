#!/bin/sh

MINICONDA_PATH=/goinfre/miniconda
INSTALL=1
SCRIPT=Miniconda3-latest-MacOSX-x86_64.sh

if [ -d "$MINICONDA_PATH" ]; then
	DEFAULT="yes"
	echo -n "Download and reinstall 64-bit Miniconda? [yes|no]
[$DEFAULT] >>> "
	read ans
	if [[ $ans == "" ]]; then
		ans=$DEFAULT
	fi
	if [[ ($ans != "yes") && ($ans != "Yes") && ($ans != "YES") && ($ans != "y") && ($ans != "Y")]]; then
		echo "
Skipping install of miniconda...
"
		INSTALL=0
	else
		rm -rf $MINICONDA_PATH
		INSTALL=1
	fi
fi

if [ $INSTALL == 1 ]; then
	curl -LO http://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
	sh $SCRIPT -b -p $MINICONDA_PATH
	rm $SCRIPT
	conda install --yes --file requirements.txt
	conda install --yes -c conda-forge tsfresh xgboost imbalanced-learn umap-learn
fi

