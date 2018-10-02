import argparse
import sys
import nnutils
import json

def main():
	cli = argparse.ArgumentParser(description='Predict the Image')

	cli.add_argument('--image_file', 
						action='store',
						default='./flowers/test/1/image_06752.jpg',
						help='The base directory where the data files are available.')
    
	cli.add_argument('--check_point',
						action='store',
						default='./checkpoint.pth',
						help='The base directory where the data files are available.')
    
	cli.add_argument('--category_names', action='store',
						default='cat_to_name.json',
	                    dest='category_names',
	                    help='File with the category names.')

	cli.add_argument('--top_k', action='store',
						default=5,
	                    dest='top_k',
	                    type=int,
	                    help='How many Top Probabilities to be displayed')

	cli.add_argument('--gpu', action='store',
						default='gpu',
						help='If provided the GPU will be used for training the model.')

	args = cli.parse_args()

	image_file = args.image_file
	check_point = args.check_point
	category_names = args.category_names
	top_k = args.top_k
	processing = args.gpu
    
	model = nnutils.load_checkpoint(check_point)

	with open(category_names, 'r') as json_file:
	    cat_to_name = json.load(json_file)
        
	top_probs, top_classes = nnutils.predict(image_file, model, top_k, processing)

	for idx, val in enumerate(top_classes):
	    print(f"Probability of the flower being {cat_to_name[val]} is {top_probs[idx]*100:.2f}.")
if __name__ == '__main__':
	main()
