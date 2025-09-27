# Train with MuJoCo (Domain Randomization ON by default)

export NO_DR=0 # or just don't set NO_DR at all
export PHYSICS=mujoco
python -u train.py 2>&1 | tee -a terminal.log

# Train with PyBullet (GUI ON, Domain Randomization OFF)

export PHYSICS=bullet
export NO_DR=1
export WITH_GUI=1
python -u train.py 2>&1 | tee -a terminal.log
