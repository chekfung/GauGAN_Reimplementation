from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import sys

def plot(data_dir): 
	# Read Dataset
	df = pd.read_csv(data_dir)
	# Plot FIDs
	sns.lineplot(x='Epoch Num', y='Average FID', data=df)
	plt.show()
	# Plot generator and discriminator loss over time
	sns.lineplot(x='Epoch Num', y='Average Generator Loss', data=df)
	plt.show()
	sns.lineplot(x='Epoch Num', y='Average Discriminator Loss', data=df)
	plt.show()

def main(data_dir): 
	plot(data_dir)

if __name__ == '__main__':
	data_dir = 'logs/fid_losses_train.csv' if len(sys.argv) == 1 else sys.argv[1]
	main(data_dir) 
