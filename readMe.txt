ReadMe For Documentation of Analysis:

1.) Install Github
2.) Install VS Code
	a.) Get Python3 on VS Code
	a.) easy-to-use addons:
		I. Jupyter Notbooke
		II. IPython
		III. Neuron (enables interactive window and MATLAB-style visualization of code blocks)
3.) Install Dependencies:
	a.) suite2p
	b.) ScanImageTiffReader
	c.) Numpy
	d.) matplotlib
4.) Open:
	a.) kd_suite2p_use_old_rois3.py in vsCode
	b.) bci_session_summary_plots.py
	c.) find_a_conditioned_neuron.py
5.) Day1 BCI Analysis
	a.) run "a" from step 4, but make sure you only specify "folder" in beggining, 
		and comment out any "old_folder" variables
		I. This is an image registration step that allows suite2p to 
			extract ROIs from fov tiff stack that was acquired during 
			experiment
	b.) After running, make sure behavior numpy array is in experiment data folder
		(the *date*-bpod_zaber.npy file)
	c.) Run "b" from step 4 with correct file specified
		I. This will generate summary of session
	d.) Run "c" from step 4
		I. This will generate conditioned neuron candidates for next experiment
6.) Day1+n BCI Analysis
	a.) same as step 5, but this time in "5a" you specify "old_folder" variable to be the folder
		that contains the suite2p roi extraction was obtained from the Day1 analysis
		I. This "old_folder" uncommenting occurs in "5b" as well.
	b.) run this pipeline to generate basic summary analysis and further generate a list of
		more candidate cns