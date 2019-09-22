# Deep-Reinforcement-Learning-for-5G-Networks

1- Check the value of self.M_ULA in `environment.py`.  Then create a subfolder with that value and run the `main.py` file.  For example `figures_M=4` if `self.M_ULA = 4`.  Move the measurement_x_seedy.txt files to that folder.

2- Change self.M_ULA to the next value (I did 4, 8, 16, 32, 64).  Then create another subfolder with that name: `figures_M=8`, ...  Move the measurements to this new folder.

3- Repeat step 2 until you are done with the number of M's.

4- Copy parse_new.py to every one of these figures_M=k folders.  Then run it from the shell in from these folders: `./parse_new.py`.  This will generate 4 files in each of these folders.

5- Make sure a folder 'figures' is created.  Change line 20 in `plotting_v5.py` to reflect the path where your 'figures' folder is.  Now run `plotting_v5.py` and you should be good.
