import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as data
from PIL import Image, ImageFile
from pathlib import Path
from torchvision import transforms
from torch.utils.data.sampler import BatchSampler
import Block_version_5
import PIL
import matplotlib.pyplot as plt
import os
#import argparse
from dataloader import *
#import math
import sys


#def count_parameters(model):
#    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
inference_mode = 1
n_epochs = 3505
learn_T1 = 0.0001
learn_R1 = 0.00001

learn_S = 0.0001
batch_size = 1
b1 = 0.5 ; b2 = 0.999 

n_threads = 0

root = r'C:\Users\Admin\Desktop\Reflection_removal'    


L1_loss = torch.nn.L1Loss()
Reconstruct_loss = torch.nn.L1Loss()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

load_student_state_path = r'.\Weighted\student_loss120_ver4'
load_teacher_T_state_path = None
load_teacher_R_state_path = None

teacher_T = Block_version_5.T_teacher_net()
teacher_T = nn.DataParallel(teacher_T)

teacher_R = Block_version_5.R_teacher_net()
teacher_R = nn.DataParallel(teacher_R)

student = Block_version_5.student_net()
student = nn.DataParallel(student)

if inference_mode:  

    batch_size = 1 
    
    pth = torch.load(load_student_state_path) ;  student.load_state_dict(pth) 
    
    teacher_T.eval() ; teacher_T.to(device)

    teacher_R.eval() ; teacher_R.to(device)
   
    student.eval() ; student.to(device)
    
else:
    teacher_T.train() ; teacher_T.to(device)
    
    teacher_R.train() ; teacher_R.to(device)

    student.train() ; student.to(device)


if load_teacher_T_state_path: 
    pth = torch.load(load_teacher_T_state_path) ; teacher_T.load_state_dict(pth) 
if load_teacher_R_state_path: 
    pth = torch.load(load_teacher_R_state_path) ; teacher_R.load_state_dict(pth)


train_dataset = datasets( root , 'Test' , 'Test', transform ) # instantiate  dataset
train_dataloader = DataLoader(train_dataset, batch_size = batch_size , shuffle = False , num_workers=0 )   

optimizer_teacher_T = torch.optim.Adam(teacher_T.parameters(), lr = learn_T1, betas=(b1, b2))
optimizer_teacher_R = torch.optim.Adam(teacher_R.parameters(), lr = learn_R1, betas=(b1, b2))
optimizer_student = torch.optim.Adam(student.parameters(), lr = learn_S, betas=(b1, b2))

if inference_mode == None:
    for epoch in range(n_epochs):
        
        for batch_idx , (reflect_image , clean_image , _ , _ , _ , _) in enumerate(train_dataloader):
            
            optimizer_teacher_T.zero_grad()
            optimizer_teacher_R.zero_grad()
            optimizer_student.zero_grad()
            
            
            reflect_image = reflect_image.to(device)
            clean_image = clean_image.to(device)
            
            compoment_image = reflect_image - clean_image  
            
            teacher_T_out1,  teacher_T_out2 , teacher_T_out3, teacher_T_out4 , \
            teacher_T_out5,  teacher_T_out6 , teacher_T_out7, teacher_T_out8 , \
            teacher_T_out = teacher_T( clean_image )
            
            teacher_T_loss = L1_loss( teacher_T_out , clean_image )
            
            teacher_R_out1,  teacher_R_out2 , teacher_R_out3 , teacher_R_out4  , \
            teacher_R_out5,  teacher_R_out6 , teacher_R_out7 , teacher_R_out8 , \
            teacher_R_out = teacher_R( compoment_image )
            
            teacher_R_loss = L1_loss( teacher_R_out , compoment_image )
            
            sout_share1,  sout_share2 , sout_share3 , sout_share4 , \
            sout_T1 , sout_T2 , sout_T3 , sout_T4 , \
            sout_R1 , sout_R2 , sout_R3 , sout_R4 , \
            student_T_out , student_R_out , \
            att1 , att2 , att3 , att4 = student( reflect_image )
        
           
            RM_T_loss_1 = L1_loss( teacher_T_out1.detach() , sout_share1[:,64:,:,:] ) ; RM_T_loss_2 = L1_loss( teacher_T_out2.detach() , sout_share2[:,64:,:,:] ) 
            RM_T_loss_3 = L1_loss( teacher_T_out3.detach() , sout_share3[:,64:,:,:]  ) ; RM_T_loss_4 = L1_loss( teacher_T_out4.detach() ,sout_share4[:,64:,:,:] ) 
            RM_T_loss_5 = L1_loss( teacher_T_out5.detach() , sout_T1 ) ;  RM_T_loss_6 = L1_loss( teacher_T_out6.detach() , sout_T2 ) ; 
            RM_T_loss_7 = L1_loss( teacher_T_out7.detach() , sout_T3 ) ;  RM_T_loss_8 = L1_loss( teacher_T_out8.detach() , sout_T4 ) ;
            
            RM_R_loss_1 = L1_loss( teacher_R_out1.detach() , sout_share1[:,:64,:,:] ) ; RM_R_loss_2 = L1_loss( teacher_R_out2.detach() , sout_share2[:,:64,:,:] ) 
            RM_R_loss_3 = L1_loss( teacher_R_out3.detach() , sout_share3[:,:64,:,:] ) ; RM_R_loss_4 = L1_loss( teacher_R_out4.detach() , sout_share4[:,:64,:,:] ) 
            RM_R_loss_5 = L1_loss( teacher_R_out5.detach() , sout_R1 ) ; RM_R_loss_6 = L1_loss( teacher_R_out6.detach() , sout_R2 ) 
            RM_R_loss_7 = L1_loss( teacher_R_out7.detach() , sout_R3 ) ; RM_R_loss_8 = L1_loss( teacher_R_out8.detach() , sout_R4 ) 
            

            RM_Top =  RM_T_loss_1 +  RM_T_loss_2 +  RM_T_loss_3 + RM_T_loss_4 + RM_T_loss_5 + RM_T_loss_6 + RM_T_loss_7 + RM_T_loss_8
            
            RM_Bot =  RM_R_loss_1 +  RM_R_loss_2 +  RM_R_loss_3 + RM_R_loss_4 + RM_R_loss_5 + RM_R_loss_6 + RM_R_loss_7 + RM_R_loss_8
                         
            I_T_loss = L1_loss( reflect_image - teacher_T_out , teacher_R_out)
            
            I_Tout_loss = L1_loss( reflect_image - teacher_T_out , student_R_out )
            
            I_OUT_T_loss = L1_loss( reflect_image - student_T_out , teacher_R_out )
            
            Res_loss = Reconstruct_loss(student_R_out , compoment_image ) + Reconstruct_loss(student_T_out , clean_image ) + \
            0.3 * (  RM_Bot )  + 0.3 * ( RM_Top )  + 0.3 * I_OUT_T_loss + 0.3 * I_T_loss + 0.3 * I_Tout_loss 
           
            student_loss = Res_loss  
            teacher_T_loss.backward(retain_graph = True ) ; teacher_R_loss.backward(retain_graph = True )
           
            student_loss.backward(retain_graph = True )
            optimizer_teacher_T.step() ; optimizer_teacher_R.step() #; optimizer_teacher_I.step()
            optimizer_student.step()
              
else:
    #total_loss = 0.0 
    #lpips_loss = 0.0
    
    for epoch in range(1):

        for batch_idx , (reflect_image , clean_image , name , h , w , c) in enumerate(train_dataloader):
            with torch.no_grad():
            
                reflect_image = reflect_image.to(device)
                clean_image = clean_image.to(device)
                
                compoment_image = reflect_image - clean_image  
                
                sout_share1,  sout_share2 , sout_share3 , sout_share4 , \
                sout_T1 , sout_T2 , sout_T3 , sout_T4 , \
                sout_R1 , sout_R2 , sout_R3 , sout_R4 , \
                student_T_out , student_R_out = student( reflect_image )
     
                #loss_fn = lpips.LPIPS(net='alex')
                
                #dis = loss_fn.forward(clean_image.cpu() , student_T_out.detach().cpu())
                
                #lpips_loss = lpips_loss + dis 
                
                #h , w = student_T_out.detach()[0,:,:,:].cpu().numpy()
              
                batch_idx_X = batch_idx + 1
                #----------------------------------------------------------------------------------show the same size
                im = transforms.ToPILImage()(student_T_out[0,:,:,:].cpu()).convert('RGB')
                x = im.resize((w,h),Image.ANTIALIAS) 
                plt.imsave(rf'C:\Users\Admin\Desktop\Reflection_removal\Output\{batch_idx_X}.png' , np.array(x)   )
                
                #plt.imsave(rf'C:\Users\Admin\Desktop\Distilling_from_TWCC\Distilling\save_re\{batch_idx_X}.png' , np.transpose(
                #        np.clip(reflect_image.detach()[0,:,:,:].cpu().numpy(), 0, 1) , ( 1 , 2 , 0) ) ) 
                
                #plt.imsave(rf'C:\Users\Admin\Desktop\Distilling_from_TWCC\Distilling\clean\{batch_idx_X}.png' , np.transpose(
                #        np.clip(clean_image.detach()[0,:,:,:].cpu().numpy(), 0, 1) , ( 1 , 2 , 0) ) ) 
          
     
        print("ok") 
  
    #print(f'loss : {lpips_loss / 200    }')












