import argparse
import numpy as np

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 
from mimic3models.preprocessing import Discretizer, Normalizer


"""parser = argparse.ArgumentParser(
	description='data loader for the MIMIC-III or PhysioNet 2012 tasks')
parser.add_argument('--path', type=str, help='location of the data')
parser.add_argument('--task', type=str, 
	help='the dataset to be loaded (mimic_mortality, physionet_mortality)')

args = parser.parse_args()"""

def mimic_loader(task='mortality',data_percentage=100):
	if task == 'mortality':
		
		print('loading mimic-iii in-hospital mortality dataset')

		from mimic3models.in_hospital_mortality import utils
		from mimic3benchmark.readers import InHospitalMortalityReader

		train_reader = InHospitalMortalityReader(dataset_dir='../data/in-hospital-mortality/train',
                                         listfile='../data/in-hospital-mortality/train_listfile.csv',
                                         period_length=48.0)
		val_reader = InHospitalMortalityReader(dataset_dir='../data/in-hospital-mortality/train',
		                                       listfile='../data/in-hospital-mortality/val_listfile.csv',
		                                       period_length=48.0)
		test_reader = InHospitalMortalityReader(dataset_dir='../data/in-hospital-mortality/test',
		                                            listfile='../data/in-hospital-mortality/test_listfile.csv',
		                                            period_length=48.0)


		discretizer = Discretizer(timestep=float(1.0),
		                          store_masks=True,
		                          impute_strategy='previous',
		                          start_time='zero')

		discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')
		cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

		normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
		normalizer_state = None
		if normalizer_state is None:
		    normalizer_state = 'ihm_ts{}.input_str:{}.start_time:zero.normalizer'.format(1.0, 'previous')
		    normalizer_state = os.path.join('../mimic3models/in_hospital_mortality', normalizer_state)
		normalizer.load_params(normalizer_state)

		headers = ['Capillary refill rate->0.0', 'Capillary refill rate->1.0', 'Diastolic blood pressure', 
		                        'Fraction inspired oxygen', 'Glascow coma scale eye opening->To Pain', 
		                        'Glascow coma scale eye opening->3 To speech', 'Glascow coma scale eye opening->1 No Response', 
		                        'Glascow coma scale eye opening->4 Spontaneously', 'Glascow coma scale eye opening->None', 
		                        'Glascow coma scale eye opening->To Speech', 'Glascow coma scale eye opening->Spontaneously', 
		                        'Glascow coma scale eye opening->2 To pain', 'Glascow coma scale motor response->1 No Response', 
		                        'Glascow coma scale motor response->3 Abnorm flexion', 'Glascow coma scale motor response->Abnormal extension', 
		                        'Glascow coma scale motor response->No response', 'Glascow coma scale motor response->4 Flex-withdraws', 
		                        'Glascow coma scale motor response->Localizes Pain', 'Glascow coma scale motor response->Flex-withdraws', 
		                        'Glascow coma scale motor response->Obeys Commands', 'Glascow coma scale motor response->Abnormal Flexion', 
		                        'Glascow coma scale motor response->6 Obeys Commands', 'Glascow coma scale motor response->5 Localizes Pain', 
		                        'Glascow coma scale motor response->2 Abnorm extensn', 'Glascow coma scale total->11', 
		                        'Glascow coma scale total->10', 'Glascow coma scale total->13', 'Glascow coma scale total->12', 
		                        'Glascow coma scale total->15', 'Glascow coma scale total->14', 'Glascow coma scale total->3', 
		                        'Glascow coma scale total->5', 'Glascow coma scale total->4', 'Glascow coma scale total->7', 
		                        'Glascow coma scale total->6', 'Glascow coma scale total->9', 'Glascow coma scale total->8', 
		                        'Glascow coma scale verbal response->1 No Response', 'Glascow coma scale verbal response->No Response', 
		                        'Glascow coma scale verbal response->Confused', 'Glascow coma scale verbal response->Inappropriate Words', 
		                        'Glascow coma scale verbal response->Oriented', 'Glascow coma scale verbal response->No Response-ETT', 
		                        'Glascow coma scale verbal response->5 Oriented', 'Glascow coma scale verbal response->Incomprehensible sounds', 
		                        'Glascow coma scale verbal response->1.0 ET/Trach', 'Glascow coma scale verbal response->4 Confused', 
		                        'Glascow coma scale verbal response->2 Incomp sounds', 'Glascow coma scale verbal response->3 Inapprop words', 
		                        'Glucose', 'Heart Rate', 'Height', 'Mean blood pressure', 'Oxygen saturation', 'Respiratory rate', 
		                        'Systolic blood pressure', 'Temperature', 'Weight', 'pH', 'mask->Capillary refill rate', 
		                        'mask->Diastolic blood pressure', 'mask->Fraction inspired oxygen', 'mask->Glascow coma scale eye opening', 
		                        'mask->Glascow coma scale motor response', 'mask->Glascow coma scale total', 
		                        'mask->Glascow coma scale verbal response', 'mask->Glucose', 'mask->Heart Rate', 'mask->Height', 
		                        'mask->Mean blood pressure', 'mask->Oxygen saturation', 'mask->Respiratory rate', 'mask->Systolic blood pressure', 
		                        'mask->Temperature', 'mask->Weight', 'mask->pH']

		print('start loading the data')

		if data_percentage != 100: # accepted values: [10,20,30,40,50,60,70,80,90]
			print('loading the partially covered testing data')
			test_reader = InHospitalMortalityReader(dataset_dir='../data/in-hospital-mortality/test_'+str(data_percentage),
		                                            listfile='../data/in-hospital-mortality/test_listfile.csv',
		                                            period_length=48.0)
			test_raw = utils.load_data(test_reader, discretizer, normalizer, False)
			x_test = np.copy(test_raw[0])
			return x_test

		# Read data
		train_raw = utils.load_data(train_reader, discretizer, normalizer, False)
		val_raw = utils.load_data(val_reader, discretizer, normalizer, False)
		test_raw = utils.load_data(test_reader, discretizer, normalizer, False)

		print('finish loading the data, spliting train, val, and test set')

		## train and validation data

		x_train = np.copy(train_raw[0])
		y_train = np.zeros((len(train_raw[1]),2))
		y_train[:,1] = np.array(train_raw[1])
		y_train[:,0] = 1 - y_train[:,1]

		x_val = np.copy(val_raw[0])
		y_val = np.zeros((len(val_raw[1]),2))
		y_val[:,1] = np.array(val_raw[1])
		y_val[:,0] = 1 - y_val[:,1]

		x_test = np.copy(test_raw[0])
		y_test = np.zeros((len(test_raw[1]),2))
		y_test[:,1] = np.array(test_raw[1])
		y_test[:,0] = 1 - y_test[:,1]

	return [x_train, x_val, x_test, y_train, y_val, y_test]

def physionet_loader(task='mortality', scale=True, input_path='../'):

	input_folder = input_path+'physionet_data_a/'

	x_train = np.load(input_folder+'1_train_x.npy')
	x_val = np.load(input_folder+'1_val_x.npy')
	x_test = np.load(input_folder+'1_test_x.npy')

	y_train = np.zeros((len(x_train),2))
	y_val = np.zeros((len(x_val),2))
	y_test = np.zeros((len(x_test),2))

	if task == 'mortality':

		y_folder = input_folder+'physionet_mortality_label/'

	elif task == 'los':

		y_folder = input_folder+'physionet_los_label/'

	y_train[:,1] = np.load(y_folder+'1_train_y.npy').reshape(-1,)
	y_val[:,1] = np.load(y_folder+'1_val_y.npy').reshape(-1,)
	y_test[:,1] = np.load(y_folder+'1_test_y.npy').reshape(-1,)
	y_train[:,0] = 1 - y_train[:,1]
	y_val[:,0] = 1 - y_val[:,1]
	y_test[:,0] = 1 - y_test[:,1]

	if scale:
		from sklearn.preprocessing import StandardScaler
		scalers = {}
		for i in range(x_train.shape[2]):
		    scalers[i] = StandardScaler()
		    x_train[:,:,i] = scalers[i].fit_transform(x_train[:,:,i])
		    
		for i in range(x_val.shape[2]):
		    x_val[:,:,i] = scalers[i].transform(x_val[:,:,i])
		    x_test[:,:,i] = scalers[i].transform(x_test[:,:,i])

	return [x_train, x_val, x_test, y_train, y_val, y_test]