# Train with MuJoCo (Domain Randomization ON by default)

export DR=1
export PHYSICS=mujoco
python -u train.py 2>&1 | tee -a terminal.log

# Train with PyBullet (GUI ON, Domain Randomization OFF)

export PHYSICS=bullet
export DR=0
export WITH_GUI=1
python -u train.py 2>&1 | tee -a terminal.log
