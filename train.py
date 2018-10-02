import argparse
import sys
import nnutils

def main():
	cli = argparse.ArgumentParser(description='Train The Image Classifier Network')

	cli.add_argument('data_dir', 
						action='store',
						help="The base directory where the data files are available.")

	cli.add_argument('--arch', action='store',
						default='vgg19',
	                    dest='arch',
	                    help='Which Pre Trained Model to be used.')

	cli.add_argument('--epochs', action='store',
						default=5,
	                    dest='epochs',
	                    type=int,
	                    help='Number of epochs to train the model.')

	cli.add_argument('--lr', action='store',
						default=0.001,
	                    dest='learning_rate',
	                    help='Learning Rate to be used.')

	cli.add_argument('--drop', action='store',
						default=0.25,
	                    dest='dropout',
	                    help='Dropout to be used in the model.')

	cli.add_argument('--save', action='store',
						default='checkpoint.pth',
	                    dest='save_dir',
	                    help='Where to save the check point.')

	cli.add_argument('--gpu', action='store',
						default='gpu',
						help='If provided the GPU will be used for training the model.')

	cli.add_argument('--hidden_units', action='store',
						default=2000,
						dest='hidden_units',
						type=int,
						help='Hideen Layer 1 nodes.')

	args = cli.parse_args()

	read_location = args.data_dir
	save_location = args.save_dir
	learning_rate = args.learning_rate
	epochs = args.epochs
	dropout = args.dropout
	arch = args.arch
	processing = args.gpu
	hidden_units = args.hidden_units
    
	print("File Location : " + read_location)
	print("save location :" + save_location)    
	dataloaders, datasets = nnutils.transform_load_data(read_location)

	model, optimizer, criterion, classifier = nnutils.create_nn(arch, dropout, learning_rate, hidden_units, processing)

	nnutils.train_model(model, criterion, optimizer, dataloaders, epochs, 25, processing)

	nnutils.save_checkpoint(model, classifier, optimizer, epochs, datasets['train'],save_location)


if __name__ == '__main__':
	main()
