# +
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

from PIL import Image
from torchvision import  transforms as T
import os
import cv2
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# -

transform = T.Compose([
    T.Resize([256, 256]),
    T.ToTensor(),
    ])

class datasets( Dataset ): 
    def __init__(self, root, img, Gt_img, transforms = None) :
        self.root_dir = root 
        self.img_folder = img
        self.Gt_img_folder = Gt_img
        
        self.path = os.path.join( self.root_dir , self.img_folder )
        self.img_path = os.listdir( self.path )
        #self.img_path = sort(self.img_path)
        
        self.Gt_path = os.path.join( self.root_dir , self.Gt_img_folder )
        self.Gt_img_path = os.listdir( self.Gt_path )
        #self.Gt_img_path = sort(self.Gt_path)
        #self.Gt_img_path.sort( key = lambda x:int( x[1:-4]) )
        #self.img_path.sort( key = lambda x:int( x[1:-4]) )
        self.Gt_img_path.sort(  )
        self.img_path.sort(  )
        self.transforms = transforms
      
        
        self.images = []
        self.Gt_images = []
        
        for f_name in tqdm(self.img_path):
            with open( os.path.join(self.path , f_name ) , "rb") as f:
                self.images.append(  np.array( Image.open( f ).convert('RGB') ) )
        for f_name in tqdm(self.Gt_img_path):
            with open( os.path.join(self.Gt_path , f_name ) , "rb") as f:
                self.Gt_images.append(  np.array( Image.open( f ).convert('RGB') ) )
        
        #for f_name in tqdm(self.Gt_img_path):
          #  with open( os.path.join(self.Gt_path , f_name ) , "rb") as f:
           #    self.Gt_images.append(  np.asarray( bytearray( f.read() ) , dtype = "uint8"   ))  
       # Gt_img = Image.open( img_Gt_path_name ).convert('RGB'))
       # for f_name in tqdm(self.Gt_img_path) :
        #    with open(os.path.join(self.Gt_path , f_name ) , "rb") as f:
         #        self.Gt_images.append( f.read() )
        #self.images = np.array(self.images)
       # self.Gt_images = np.array(self.Gt_images)
    def __getitem__(self, idx ):

       # img_name = self.img_path[ idx ]
        img_Gt_name = self.Gt_img_path[ idx ]

       # img_input_path_name = os.path.join( self.root_dir , self.img_folder , img_name )
       # img_Gt_path_name = os.path.join( self.root_dir , self.Gt_img_folder , img_Gt_name )
        
      
        
      #  img = Image.open( img_input_path_name ).convert('RGB')
       # Gt_img = Image.open( img_Gt_path_name ).convert('RGB')
        
        img = Image.fromarray(np.uint8(self.images[idx])) 
        
        Gt_img = Image.fromarray(np.uint8(self.Gt_images[idx])) 
        
      
        if self.transforms:
            img = self.transforms(img)
            Gt_img = self.transforms(Gt_img)
    
        return img , Gt_img , img_Gt_name
        
    def __len__(self):
        return len(self.img_path)

# +
#root = r'/home/stitch0312/Distilling/Distilling/DSLR/train250' 

# +
#train_dataset = datasets( root , 'ERRNET_Train_I' , 'ERRNET_Train_T', transform ) # instantiate  dataset
#train_dataloader = DataLoader(train_dataset, batch_size = 4 , shuffle = False , num_workers=0 )   
# +
#plt.imshow(np.transpose( np.clip( train_dataset[460][0].numpy() , 0 , 1 ) , (1, 2, 0 ) ))
# +
#plt.imshow(np.transpose( np.clip( train_dataset[460][1].numpy() , 0 , 1 ) , (1, 2, 0 ) ))
# -




