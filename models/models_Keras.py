import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
import os 
import sys
import numpy as np
import keras
from keras import metrics
from keras.models import *
from keras.engine import *
from keras.optimizers import *
from keras.callbacks import *
from keras import backend as keras
from keras import backend as K

sys.path.append('../data_manipulation')
sys.path.append('../models')
sys.path.append('../utils')
"""
train the models with patchs or full images
"""
from data_generator import *
from data_Load import *
from losses import *
from metrics import *
import models_Keras
from models_load import *
import pandas as pd
	
#train with patch, using generator for training set and loading every single patch of one image for validation 
def train_segAn(patch_size,batch_size,num_epoch,steps,output_path,niter_disc=1):
	
	
	lr_seg_pre = 0.001
	lr_disc = 0.0005
	lr_seg = 0.001
	combined_losses = [logcosh_disc,dice_coef_loss]  ## [out_mae,dice_coef_loss]
	weights_mae  = 1.
	weights_dice = 1.
	combined_weights = [weights_mae , weights_dice] ##weights of the L1 (mean absolute error) and dice losses
	
	model = models_Keras.Models(batch_size = batch_size,patch_size = patch_size,spatialdrop=1)
	my_generator = Generator(patch_len=patch_size , batch_size=batch_size) #generate train images randomly
	my_loader = Load(patch_len = patch_size ,batch_size = batch_size)
	valIm,valGt = my_loader.load_val() # predict with all image at each time
	
	sess = tf.Session() # init tensorflow session for Dice calculations in validation
	
	## place holders for the inputs
	ground_truth_patch = Input(batch_shape=(batch_size,patch_size,patch_size,patch_size,1))
	image_patch = Input(batch_shape=(batch_size,patch_size,patch_size,patch_size,1))
	
	#optimizers
	opt = Adam(lr = lr_seg, beta_1=0.5)
	dopt = Adam(lr = lr_disc,beta_1=0.5)
	
	# get models for discriminator and segmentor, the fixed models are Networks(keras class) objects
	discriminator,discriminator_fixed = model.get_discriminator()
	segmentor,segmentor_fixed = model.get_segmentor()

	# Debug: discriminator and segmentor weights
	n_disc_trainable = len(discriminator.trainable_weights)
	n_seg_trainable = len(segmentor.trainable_weights)
	
	##model that trains the discriminator maximizing the MAE loss
	segmentor_fixed.trainable = False
	predictions_patch_fixed = segmentor_fixed(image_patch)
	output_error_disc = discriminator([image_patch,predictions_patch_fixed,ground_truth_patch])
	combined_discriminator = Model(inputs=[image_patch,ground_truth_patch],outputs = output_error_disc)
	combined_discriminator.compile(optimizer = dopt, loss = neg_logcosh_disc) ##neg_out_mae
	
	
	##model that train the segmentor minimizing the MAE loss and the dice loss from the predictions
	discriminator_fixed.trainable = False
	predictions_patch = segmentor(image_patch)
	output_error_seg = discriminator_fixed([image_patch,predictions_patch,ground_truth_patch])
	combined_segmentor = Model(inputs=[image_patch,ground_truth_patch],outputs = [output_error_seg,predictions_patch])
	combined_segmentor.compile(optimizer = opt, loss = combined_losses, loss_weights = combined_weights, metrics = combined_losses)
	
	
	output_fake = np.zeros(combined_segmentor.output_shape[0])

	
	# Debug: compare if trainable weights correct
	assert(len(combined_discriminator._collected_trainable_weights) == n_disc_trainable)
	assert(len(combined_segmentor._collected_trainable_weights) == n_seg_trainable)

	print 'assert before prefit'
	## Segmentor pretraining
	
	#segmentor.compile(optimizer = Adam(lr = lr_seg_pre,amsgrad=True), loss = dice_coef_loss)
	#print 'Segmentor Pretraining'
	#history = segmentor.fit_generator(generator= my_generator.generatorRandomPatchs('train'),steps_per_epoch= 100, epochs = 50)
	

	assert(len(combined_discriminator._collected_trainable_weights) == n_disc_trainable)
	assert(len(combined_segmentor._collected_trainable_weights) == n_seg_trainable)
	print 'assert after prefit'

	## GAN training
	dice_val_init = 0
	step = 0
	epoch = 0
	dice_train = []
	avg_loss_disc = []
	avg_loss_seg = []
	avg_l1_seg = []
	
	'''csv file
	 epoch,dice train ,dice_val, l1_segmentor, loss_seg, lossl1_disc ,
	 lr_seg, lr_disc, lr_combined, weights_mae , weights_dice, batch_size, niter_disc
	'''
	metrics_names = ['epoch','dice_loss_train' ,'dice_coef_val', 'l1_segmentor', 'loss_seg', 'lossl1_disc']
	params_names = ['lr_seg', 'lr_disc', 'lr_seg_pre', 'weights_mae ', 'weights_dice', 'batch_size', 'niter_disc']
	metrics = [[],[],[],[],[],[]] 
	params = [[lr_seg] , [lr_disc] , [lr_seg_pre] , [weights_mae] , [weights_dice] , [batch_size] , [niter_disc]]
	
	save_csv(params,params_names,output_path+'_params')

	for batch_imgs, batch_gt in my_generator.generatorRandomPatchs('train'):
		step += 1
		
		## train the discrimator niter times before training the segmentor
		## Howover, the inverse should be done to improve stability, meaning that one should train the segmentor a n times more than the discriminator(critic)
		loss_discriminator = combined_discriminator.train_on_batch(x = [batch_imgs,batch_gt], y = output_fake)
		#if step%niter_disc == 0 or step == 1:
		loss_segmentor = combined_segmentor.train_on_batch(x = [batch_imgs, batch_gt],y = [output_fake,batch_gt])
		
		avg_loss_disc.append(loss_discriminator)
		avg_loss_seg.append(loss_segmentor[0])
		avg_l1_seg.append(loss_segmentor[1])
		dice_train.append(loss_segmentor[2])
		
		sys.stdout.write(' step %d / %d ; Discriminator: loss %.5f  ; combined: loss %.5f mae %.5f dice %.5f ; train avg_dice %.5f \r'% 
		(step,steps, loss_discriminator,loss_segmentor[0],loss_segmentor[1],loss_segmentor[2],np.asarray(dice_train).mean()))
		sys.stdout.flush()

		if step == steps:
			step = 0
			
			print ''
			val_prediction = segmentor.predict(valIm,batch_size,1)
			dice_val = sess.run(dice_coef(val_prediction,valGt,0))
			print 'dice_val',dice_val
		
			if dice_val>dice_val_init:
				files_updater(output_path,segmentor,epoch,dice_val)
				dice_val_init = dice_val
			
			metrics[0].append(epoch)
			metrics[1].append(np.asarray(dice_train).mean())
			metrics[2].append(dice_val)
			metrics[3].append(np.asarray(avg_l1_seg).mean())
			metrics[4].append(np.asarray(avg_loss_seg).mean())
			metrics[5].append(np.asarray(avg_loss_disc).mean())

			save_csv(metrics,metrics_names,output_path)
			
			'''
			print ' lr : '
			K.set_value(combined_model.optimizer.lr,lr_finder(epoch))
			K.set_value(discriminator.optimizer.lr,lr_finder(epoch))
			'''
			
			print 'epoch ',epoch,'/',num_epoch
			epoch +=1
			
			dice_train = []
			avg_loss_disc = []
			avg_loss_seg = []
			avg_l1_seg = []
			
		if epoch == num_epoch:
			break
	
def save_csv(array,name_col,folder_name):
	
	df = pd.DataFrame(dict(sorted(zip(name_col,np.asarray(array)))))
	filepath='../../my_csv/'+folder_name+'.csv'
	df.to_csv(filepath)

def files_updater(folder_name,model,epoch,dice):

	if not os.path.exists('../../my_weights/'+folder_name):
		os.makedirs('../../my_weights/'+folder_name)
	filepath='../../my_weights/'+folder_name+'/64deepVal-{0:02d}-{1:.2f}.h5'.format(epoch,dice)
	model.save(filepath)

if __name__ == '__main__':
	
	'''
	it's important to specify the GPU that one wants, otherwise, the program will use the memory of all GPUs available 
	and do processing in just one of them
	'''
	
	os.environ['CUDA_VISIBLE_DEVICES'] = '0'
	
	train_segAn(patch_size = 64,batch_size= 2,num_epoch = 500,steps=50,output_path = 'SegAn_Net11_BS2_',niter_disc = 1)
