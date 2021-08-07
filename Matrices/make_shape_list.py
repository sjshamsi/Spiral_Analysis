#Gonna import all the modules we need to make a np array of the shapes of the Maps objects for galaxies in our sample
from marvin.tools.maps import Maps
import numpy as np
import logging #we'll log galaxies that don't work

#make a log life and save any errors here
logging.basicConfig(filename='shapelist_errors.log', encoding='utf-8', level=logging.DEBUG)

paths = np.load('../Select_Threshold/available_spirals.npy', allow_pickle=True) #gz3d paths for a usegul sample we made earlier

mangaids = [x.split('/')[-1].split('_')[0] for x in paths]

count = 0 #to keep track of the script
shape_list = []

for mangaid in mangaids:
    try:
        shape = Maps(mangaid)._shape[0]
    except Exception as e:
        logging.exception('Error calling Maps object for MaNGA ID {}'.format(mangaid))
        
    if shape not in shape_list:
        shape_list.append(shape)
    
    print(count)
    count += 1
    
np.save('shape_list.npy', shape_list, allow_pickle=True) #save the shape_list