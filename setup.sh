#!/bin/bash
echo How do you connect to GitHub? If you enter your username and password when you pull a private git repository, then answer https, otherwise answer ssh.
echo \(ssh or https\)
read CONN

if [ ! $CONN == "ssh" ] && [ ! $CONN == "https" ]; then 
    echo Please specify either ssh or https
else
    module load anaconda3
    module load rh
    
    python3 -m pip install --upgrade pip
    python3 -m pip install numpy
    python3 -m pip install scipy==1.9
    python3 -m pip install matplotlib
    python3 -m pip install shapely 
    python3 -m pip install gitpython 
    python3 -m pip install statsmodels 
    python3 -m pip install multipy
    python3 -m pip install opencv-python 
    python3 -m pip install pyqt5
    python3 -m pip install ujson
    python3 -m pip install scikit-learn
    
    git pull
    python3 -m pip install . 
    cd ..

    path_line1="export CPATH=/projects/LEIFER/libraries/include:/projects/LEIFER/libraries/include/vxl/core:/projects/LEIFER/libraries/include/vxl/vcl:\$CPATH"
    path_line2="export LD_LIBRARY_PATH=/projects/LEIFER/libraries/lib:\$LD_LIBRARY_PATH"
    path_line3="export LIBRARY_PATH=/projects/LEIFER/libraries/lib:\$LIBRARY_PATH"

    if ! grep -q "$path_line1" ~/.bashrc; then
      echo "export CPATH=/projects/LEIFER/libraries/include:/projects/LEIFER/libraries/include/vxl/core:/projects/LEIFER/libraries/include/vxl/vcl:\$CPATH" >> ~/.bashrc
    fi

    if ! grep -q "$path_line2" ~/.bashrc; then
      echo "export LD_LIBRARY_PATH=/projects/LEIFER/libraries/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc
    fi

    if ! grep -q "$path_line3" ~/.bashrc; then
      echo "export LIBRARY_PATH=/projects/LEIFER/libraries/lib:\$LIBRARY_PATH" >> ~/.bashrc
    fi

    export CPATH=/projects/LEIFER/libraries/include:/projects/LEIFER/libraries/include/vxl/core:/projects/LEIFER/libraries/include/vxl/vcl:$CPATH
    export LD_LIBRARY_PATH=/projects/LEIFER/libraries/lib:$LD_LIBRARY_PATH
    export LIBRARY_PATH=/projects/LEIFER/libraries/lib:$LIBRARY_PATH
    
    FILE=multipy
    if [ -d "$FILE" ]; then
        echo "multipy already there"
    else
        git clone https://github.com/puolival/multipy.git
        cd "$FILE"
        python3 -m pip install . 
        cd ..
    fi

    FILE=mistofrutta
    if [ -d "$FILE" ]; then
        cd "$FILE"
        git pull
        python3 -m pip install . 
        cd ..
    else
        if [[ ! $CONN == "https" ]]; then
            git clone git@github.com:francescorandi/mistofrutta.git
        else
            git clone https://github.com/francescorandi/mistofrutta.git
        fi
        cd "$FILE"
        python3 -m pip install . 
        cd ..
    fi
    
    FILE=gmmreg-lw
    if [ -d "$FILE" ]; then
        cd "$FILE"
        git checkout francesco
        git pull
        module unload rh
        python3 -m pip install . 
        module load rh
        cd ..
    else
        if [[ ! $CONN == "https" ]]; then
            git clone git@github.com:leiferlab/gmmreg-lw.git
        else
            git clone https://github.com/leiferlab/gmmreg-lw.git
        fi
        cd "$FILE"
        git checkout francesco
        module unload rh
        python3 -m pip install . 
        module load rh
        cd ..
    fi
    
    FILE=savitzkygolay
    if [ -d "$FILE" ]; then
        cd "$FILE"
        git pull
        python3 -m pip install . 
        cd ..
    else
        if [[ ! $CONN == "https" ]]; then
            git clone git@github.com:leiferlab/savitzkygolay.git
        else
            git clone https://github.com/leiferlab/savitzkygolay.git
        fi
        cd "$FILE"
        python3 -m pip install . 
        cd ..
    fi

    FILE=fDLC_Neuron_ID
    if [ -d "$FILE" ]; then
        cd "$FILE"
        git pull
        python3 -m pip install . 
        cd ..
    else
        if [[ ! $CONN == "https" ]]; then
            git clone git@github.com:francescorandi/fDLC_Neuron_ID.git
        else
            git clone https://github.com/francescorandi/fDLC_Neuron_ID.git
        fi
        cd "$FILE"
        python3 -m pip install . 
        cd ..
    fi

    FILE=wormbrain
    if [ -d "$FILE" ]; then
        cd "$FILE"
        git pull
        git checkout develop
        python3 -m pip install . 
        cd ..
    else
        if [[ ! $CONN == "https" ]]; then
            git clone git@github.com:leiferlab/wormbrain.git
        else
            git clone https://github.com/leiferlab/wormbrain.git
        fi
        cd "$FILE"
        git checkout develop
        python3 -m pip install . 
        cd ..
    fi
    
    FILE=wormdatamodel
    if [ -d "$FILE" ]; then
        cd "$FILE"
        git pull
        git checkout develop
        python3 -m pip install . 
        cd ..
    else
        if [[ ! $CONN == "https" ]]; then
            git clone git@github.com:leiferlab/wormdatamodel.git
        else
            git clone https://github.com/leiferlab/wormdatamodel.git
        fi
        cd "$FILE"
        git checkout develop
        python3 -m pip install . 
        cd ..
    fi

    FILE=wormneuronsegmentation-c
    if [ -d "$FILE" ]; then
        cd "$FILE"
        git pull
        git checkout develop
        python3 -m pip install . 
        cd ..
    else
        if [[ ! $CONN == "https" ]]; then
            git clone git@github.com:leiferlab/wormneuronsegmentation-c.git
        else
            git clone https://github.com/leiferlab/wormneuronsegmentation-c.git
        fi
        cd "$FILE"
        git checkout develop
        python3 -m pip install . 
        cd ..
    fi
    
    echo -----
    echo -----
    echo -----
    echo "You NEED TO copy pumpprobe/fdr.py into the install folder of multipy (fix old Python2 commands)"
    echo "e.g. <>/.venv/lib/python3.8/site-packages/multipy if you are using a virtual environment"
    echo "or ~/.local/lib/python3.8/site-packages/multipy if you installed the modules in your home folder"
    echo -----
    echo -----
    echo -----
fi
