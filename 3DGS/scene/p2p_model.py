### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np
# from .spynet import SpyNetwork
import cv2
# from .fs_networks import ResnetBlock_Adain
# from .conv import Conv2d
import math
from einops.layers.torch import Rearrange
import os

###############################################################################
# Functions
###############################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, "weight"):
            m.weight.data.normal_(0.0, 0.002)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


# TODO: 20180929: Generator Input contains two images...
class GeneratorModel:
    def __init__(self, input_nc, output_nc, ngf, netG, n_downsample_global=3, n_blocks_global=9, n_local_enhancers=1, 
                n_blocks_local=3, norm='instance', gpu_ids=[], final=nn.Tanh(), opt=None):    
        norm_layer = get_norm_layer(norm_type=norm)
        if netG == 'global':    
            netG = GlobalGenerator(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)       
        elif netG == 'local':        
            netG = LocalEnhancer(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global,
                                n_local_enhancers, n_blocks_local, norm_layer, final=final, opt=opt)
        elif netG == 'encoder':
            netG = Encoder(input_nc, output_nc, ngf, n_downsample_global, norm_layer)
        elif netG == 'upsample':
            netG = UpsampleNet(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global,
                                n_local_enhancers, n_blocks_local, norm_layer, final=final, opt=opt)
        else:
            raise('generator not implemented!')
        print(netG)
        if len(gpu_ids) > 0:
            assert(torch.cuda.is_available())   
            netG.cuda(gpu_ids[0])
            # netG.cuda()
        netG.apply(weights_init)
        self.netG = netG.cuda()

    def train_setting(self, training_args):
        l = [
            {'params': list(self.netG.parameters()),
             'lr': training_args.position_lr_init * self.spatial_lr_scale,
             "name": "generator"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.generator_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                       lr_final=training_args.position_lr_final,
                                                       lr_delay_mult=training_args.position_lr_delay_mult,
                                                       max_steps=training_args.deform_lr_max_steps)

    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "generator/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.netG.state_dict(), os.path.join(out_weights_path, 'generator.pth'))

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "generator"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, "generator/iteration_{}/generator.pth".format(loaded_iter))
        self.netG.load_state_dict(torch.load(weights_path))

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "generator":
                lr = self.generator_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

# TODO: 20180929: Discriminator Input contains two pairs...
class DiscriminatorModel:
    def __init__(self, input_nc, ndf, n_layers_D, norm='instance', use_sigmoid=False, num_D=1, getIntermFeat=False, gpu_ids=[]):   
        # (input=30,ndf=32,n_layers_D=3,instance,false,num_D=3, True, 0)     
        norm_layer = get_norm_layer(norm_type=norm)   
        netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat) 
        # (18,32,3,instance,false,3,false)  
        print(netD)
        if len(gpu_ids) > 0:
            assert(torch.cuda.is_available())
            netD.cuda(gpu_ids[0])
        netD.apply(weights_init)
        self.netD = netD.cuda()
    
    def train_setting(self, training_args):
        l = [
            {'params': list(self.netD.parameters()),
             'lr': training_args.position_lr_init * self.spatial_lr_scale,
             "name": "discriminator"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.discriminator_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                       lr_final=training_args.position_lr_final,
                                                       lr_delay_mult=training_args.position_lr_delay_mult,
                                                       max_steps=training_args.deform_lr_max_steps)

    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "discriminator/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.netD.state_dict(), os.path.join(out_weights_path, 'discriminator.pth'))

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "discriminator"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, "discriminator/iteration_{}/discriminator.pth".format(loaded_iter))
        self.netD.load_state_dict(torch.load(weights_path))

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "discriminator":
                lr = self.discriminator_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

##############################################################################
# Losses
##############################################################################
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor.to(input.device)

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()        
        # self.vgg = Vgg19().cuda()
        self.vgg = Vgg19()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def forward(self, x, y):              
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())        
        return loss


# 20181012 Implement flow loss.
# 20181020 show flow in rgb, check

def flow2im(flow):
    flow_npy = np.array(flow.detach().cpu().numpy().transpose(1, 2, 0), np.float32)
    shape = flow_npy.shape[:-1]
    hsv = np.zeros(shape + (3,), dtype=np.uint8)
    hsv[..., 2] = 255

    mag, ang = cv2.cartToPolar(flow_npy[..., 0], flow_npy[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr 
    
class FlowLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(FlowLoss, self).__init__()
        self.flownet = SpyNetwork().cuda()
        self.criterion = nn.L1Loss()
        self.count = 0

    def forward(self, real1, real2, fake1, fake2):
        real_flow, fake_flow = self.flownet(real1, real2), self.flownet(fake1, fake2)
        # print(real_flow)
        # print(fake_flow)
        # 20181020: print flow images
        # cv2.imwrite('./spynet_flow_2/{:03d}_real_flow.png'.format(self.count), flow2im(real_flow))
        # cv2.imwrite('./spynet_flow_2/{:03d}_fake_flow.png'.format(self.count), flow2im(fake_flow))
        # self.count += 1
        aa  = self.criterion(real_flow, fake_flow)
        return aa


##############################################################################
# Generator
##############################################################################
class LocalEnhancer(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=3, n_blocks_global=9, 
                 n_local_enhancers=1, n_blocks_local=3, norm_layer=nn.BatchNorm2d, padding_type='reflect',
                 final=nn.Tanh(), opt=None):        
        super(LocalEnhancer, self).__init__()
        self.n_local_enhancers = n_local_enhancers
        
        ###### global generator model #####           
        ngf_global = ngf * (2**n_local_enhancers) # 32*2=64
        opt.times = True
        model_global = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global,
                                       n_blocks_global, norm_layer, opt=opt)#.model

        # model_global = [model_global[i] for i in range(len(model_global)-3)] # get rid of final convolution layers
        # self.model = nn.Sequential(*model_global)
        self.model = model_global

        activation = nn.ReLU(True)        

        ngf_global = ngf_global // 2
        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsample_global):
            mult = 2**i
            model += [nn.Conv2d(ngf_global * mult, ngf_global * mult * 2, kernel_size=3, stride=2, padding=1),
                    norm_layer(ngf_global * mult * 2), activation]
        self.model_before = nn.Sequential(*model)
        ###### local enhancer layers #####
        for n in range(1, n_local_enhancers+1):
            ### downsample
            ngf_global = ngf * (2**(n_local_enhancers-n))
            model_downsample = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0), 
                                norm_layer(ngf_global), nn.ReLU(True),
                                nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1), 
                                norm_layer(ngf_global * 2), nn.ReLU(True)]
            ### residual blocks
            model_upsample = []
            for i in range(n_blocks_local):
                model_upsample += [ResnetBlock(ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]

            ### upsample
            model_upsample += [nn.ConvTranspose2d(ngf_global * 2, ngf_global, kernel_size=3, stride=2, padding=1, output_padding=1), 
                               norm_layer(ngf_global), nn.ReLU(True)]

            ### final convolution
            if n == n_local_enhancers:                
                # model_upsample += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
                model_upsample += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), final]
            
            setattr(self, 'model'+str(n)+'_1', nn.Sequential(*model_downsample))
            setattr(self, 'model'+str(n)+'_2', nn.Sequential(*model_upsample))
        
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)


    # 20180929: change input style
    def forward(self, input, middle_feature=None):
        ### create input pyramid
        input_downsampled = [input]
        # input_downsampled = [torch.cat([input, params.unsqueeze(2).unsqueeze(3).repeat(1, 1, 512, 512)], dim=1)]
        for i in range(self.n_local_enhancers):
            input_downsampled.append(self.downsample(input_downsampled[-1]))

        ### output at coarest level
        output_prev = self.model(self.model_before(input_downsampled[-1]), middle_feature)
        ### build up one layer at a time
        for n_local_enhancers in range(1, self.n_local_enhancers+1):
            model_downsample = getattr(self, 'model'+str(n_local_enhancers)+'_1')
            model_upsample = getattr(self, 'model'+str(n_local_enhancers)+'_2')
            input_i = input_downsampled[self.n_local_enhancers-n_local_enhancers]            
            output_prev = model_upsample(model_downsample(input_i) + output_prev)
        return output_prev

class UpsampleNet(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=3, n_blocks_global=9, 
                 n_local_enhancers=1, n_blocks_local=3, norm_layer=nn.BatchNorm2d, padding_type='reflect',
                 final=nn.Tanh(), opt=None):
        super(UpsampleNet, self).__init__()
        self.n_local_enhancers = n_local_enhancers
        
        ###### global generator model #####           
        ngf_global = ngf * (2**n_local_enhancers) # 32*2=64 
        opt.times = False
        model_global = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global,
                                       n_blocks_global, norm_layer, opt=opt)#.model

        self.model = model_global

        ###### local enhancer layers #####
        for n in range(1, n_local_enhancers+1):
            ### downsample
            ngf_global = ngf * (2**(n_local_enhancers-n))
            model_downsample = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0), 
                                norm_layer(ngf_global), nn.ReLU(True),
                                nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1), 
                                norm_layer(ngf_global * 2), nn.ReLU(True)]
            ### residual blocks
            model_upsample = []
            for i in range(n_blocks_local):
                model_upsample += [ResnetBlock(ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]

            ### upsample
            model_upsample += [nn.ConvTranspose2d(ngf_global * 2, ngf_global, kernel_size=3, stride=2, padding=1, output_padding=1), 
                               norm_layer(ngf_global), nn.ReLU(True)]

            ### final convolution
            if n == n_local_enhancers:
                model_upsample += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), final]
            
            setattr(self, 'model'+str(n)+'_1', nn.Sequential(*model_downsample))
            setattr(self, 'model'+str(n)+'_2', nn.Sequential(*model_upsample))

    def forward(self, input_downsampled, middle_feature=None):
        if isinstance(input_downsampled, list):
            output_prev, middle_feature = self.model(input_downsampled[-1], None, middle_feature)
            ### build up one layer at a time
            for n_local_enhancers in range(1, self.n_local_enhancers+1):
                model_downsample = getattr(self, 'model'+str(n_local_enhancers)+'_1')
                model_upsample = getattr(self, 'model'+str(n_local_enhancers)+'_2')
                input_i = input_downsampled[self.n_local_enhancers-n_local_enhancers]            
                output_prev = model_upsample(model_downsample(input_i) + output_prev)
            return output_prev, middle_feature
        else:
            output_prev = self.model(input_downsampled, middle_feature)
            ### build up one layer at a time
            for n_local_enhancers in range(1, self.n_local_enhancers+1):
                model_upsample = getattr(self, 'model'+str(n_local_enhancers)+'_2')          
                output_prev = model_upsample(output_prev)
            return output_prev

class PeriodicPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, period=25, max_seq_len=600):
        super(PeriodicPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(period, d_model)
        position = torch.arange(0, period, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, period, d_model)
        repeat_num = (max_seq_len//period) + 1
        pe = pe.repeat(1, repeat_num, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# Temporal Bias, inspired by ALiBi: https://github.com/ofirpress/attention_with_linear_biases
def init_biased_mask(n_head, max_seq_len, period):
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)                   
        else:                                                 
            closest_power_of_2 = 2**math.floor(math.log2(n)) 
            return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]
    slopes = torch.Tensor(get_slopes(n_head))
    bias = torch.arange(start=0, end=max_seq_len, step=period).unsqueeze(1).repeat(1, period).view(-1) // (period)
    bias = - torch.flip(bias, dims=[0])
    alibi = torch.zeros(max_seq_len, max_seq_len)
    for i in range(max_seq_len):
        alibi[i, :i+1] = bias[-(i+1):]
    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)
    mask = (torch.triu(torch.ones(max_seq_len, max_seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask = mask.unsqueeze(0) + alibi
    return mask

# Alignment Bias
def enc_dec_mask(device, dataset, T, S):
    mask = torch.ones(T, S)
    if dataset == "BIWI":
        for i in range(T):
            mask[i, i*2:i*2+2] = 0
    elif dataset == "vocaset":
        for i in range(T):
            mask[i, i] = 0
    return (mask==1).to(device=device)

class Face3DMMOneHotFormer(nn.Module):
    fc_dim=257
    def __init__(self, args, net_recon, use_last_fc=False, init_path=None, **kwargs):
        super().__init__()
        """
        audio: (batch_size, raw_wav)
        template: (batch_size, V*3)
        vertice: (batch_size, seq_len, V*3)
        """
        # self.use_last_fc = use_last_fc
        # if net_recon not in func_dict:
        #     return NotImplementedError('network [%s] is not implemented', net_recon)
        # func, last_dim = func_dict[net_recon]
        # backbone = func(use_last_fc=use_last_fc, num_classes=self.fc_dim)
        # backbone.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        # if init_path and os.path.isfile(init_path):
        #     load_state_dict(backbone, torch.load(init_path, map_location='cpu'))
        #     print("loading init net_recon %s from %s" %(net_recon, init_path))
        # self.backbone = backbone

        self.dataset = "vocaset" # args.dataset
        # motion encoder
        args.vertice_dim = 1024 #* 16
        args.feature_dim = 1024 # 64
        args.period = 30
        self.vertice_map = nn.Linear(args.vertice_dim, args.feature_dim)
        self.coeff_map = nn.Linear(257, args.feature_dim)
        # periodic positional encoding
        self.PPE = PeriodicPositionalEncoding(args.feature_dim, period=args.period)
        # self.PPE = PositionalEncoding(args.feature_dim)
        # temporal bias
        self.biased_mask = init_biased_mask(n_head=4, max_seq_len=600, period=args.period)
        decoder_layer = nn.TransformerDecoderLayer(d_model=args.feature_dim, nhead=4, dim_feedforward=2*args.feature_dim, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        # motion decoder
        self.hidden_map = nn.Linear(args.vertice_dim, args.feature_dim)
        self.vertice_map_r = nn.Linear(args.feature_dim, args.vertice_dim)
        # style embedding
        self.obj_vector = nn.Linear(80, args.feature_dim, bias=False)

        # nn.init.constant_(self.vertice_map_r.weight, 0)
        # nn.init.constant_(self.vertice_map_r.bias, 0)

        self.config = args

        # channels = 1024
        # patch_height = 4
        # patch_width = 4
        # patch_dim = channels * patch_height * patch_width
        # self.to_patch_embedding = nn.Sequential(
        #     Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
        #     nn.Linear(patch_dim, args.feature_dim),
        # )

    def forward(self, hidden_states, pred_face=None):
        # tgt_mask: :math:`(T, T)`.
        # memory_mask: :math:`(T, S)`.
        device = hidden_states.device
        one_hot = torch.zeros((1, 80), device=device)
        one_hot[0, 0] = 1.
        obj_embedding = self.obj_vector(one_hot) # (1, feature_dim)
        frame_num = hidden_states.shape[0]
        # hidden_states = self.backbone(torch.cat([img, pred_face], dim=1))
        hidden_states = hidden_states.squeeze().unsqueeze(0)
        hidden_states = self.hidden_map(hidden_states)
        # hidden_states = self.to_patch_embedding(hidden_states)
        # hidden_states = hidden_states.reshape(-1, 1024).unsqueeze(0)

        for i in range(frame_num):
            if i == 0:
                vertice_emb = obj_embedding.unsqueeze(1) # (1, 1, feature_dim)
                style_emb = vertice_emb
                vertice_input = self.PPE(style_emb)
            else:
                vertice_input = self.PPE(vertice_emb)
            tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().to(device=device)
            # tgt_mask = None

            memory_mask = enc_dec_mask(device, self.dataset, vertice_input.shape[1], hidden_states.shape[1])
            # memory_mask = None

            x = self.transformer_decoder(vertice_input, hidden_states, tgt_mask=tgt_mask, memory_mask=memory_mask)
            vertice_out = self.vertice_map_r(x)
            new_output = self.vertice_map(vertice_out[:, -1, :]).unsqueeze(1)
            new_output = new_output + style_emb
            vertice_emb = torch.cat((vertice_emb, new_output), 1)
        
        # return vertice_out[0].reshape(frame_num, -1, 1, 1)
        # vertice_out = vertice_out[0]
        # vertice_out = vertice_out.reshape(frame_num, -1, 1024)
        # vertice_out = Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=4, p2=4)
        # return vertice_out
        return vertice_out[0]

    def predict(self, batch):
        audio = batch['raw_audio']
        # one_hot = batch['one_hot']
        id_coeff = batch['id_coeff']

        device = audio.device

        # obj_embedding = self.obj_vector(one_hot)
        obj_embedding = self.obj_vector(id_coeff)
        frame_num = int(audio.shape[1] / 16000 * 25)
        # hidden_states = self.audio_encoder(audio, self.dataset, output_fps=25).last_hidden_state
        hidden_states = self.audio_encoder(audio, self.dataset, frame_num=frame_num).last_hidden_state
        if self.dataset == "BIWI":
            frame_num = hidden_states.shape[1]//2
        elif self.dataset == "vocaset" or self.dataset == 'FLAME':
            frame_num = hidden_states.shape[1]
        audio = hidden_states
        hidden_states = self.audio_feature_map(hidden_states)

        for i in trange(frame_num):
            if i == 0:
                vertice_emb = obj_embedding.unsqueeze(1) # (1, 1, feature_dim)
                style_emb = vertice_emb
                vertice_input = self.PPE(style_emb)
            
            else:
                vertice_input = self.PPE(vertice_emb)

            tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().to(device=device)
            memory_mask = enc_dec_mask(device, self.dataset, vertice_input.shape[1], hidden_states.shape[1])
            vertice_out = self.transformer_decoder(vertice_input, hidden_states, tgt_mask=tgt_mask, memory_mask=memory_mask)
            vertice_out = self.vertice_map_r(vertice_out)
            new_output = self.vertice_map(vertice_out[:,-1,:]).unsqueeze(1)
            new_output = new_output + style_emb
            vertice_emb = torch.cat((vertice_emb, new_output), 1)
        
        return vertice_out, audio

class GlobalGenerator_copy(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d, 
                 padding_type='reflect'):
        assert(n_blocks >= 0)
        super(GlobalGenerator_copy, self).__init__()        
        activation = nn.ReLU(True)        

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                    norm_layer(ngf * mult * 2), activation]
        #     if i < 4:
        #         mult = 2**i
        #         model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
        #                 norm_layer(ngf * mult * 2), activation]
        #     else:
        #         model += [nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),
        #                 norm_layer(1024), activation]
        # model += [nn.Conv2d(1024, 1024, kernel_size=4, stride=1, padding=0),
        #             activation]
        
        # class a():
        #     self.net_recon = None
        #     self.use_last_fc = False
        #     self.init_path = ''
        #     def __init__(self) -> None:
        #         pass
        # opt = a()
        # opt.net_recon = None
        # opt.use_last_fc = False
        # opt.init_path = ''
        # # self.con_former = Face3DMMOneHotFormer(opt, 
        # #     net_recon=opt.net_recon, use_last_fc=opt.use_last_fc, init_path=opt.init_path)

        ### resnet blocks
        model_2 = list()
        
        mult = 2**n_downsampling
        
        for i in range(n_blocks):
            # model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
            model_2 += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        
        ### upsample
        # model_2 += [nn.ConvTranspose2d(1024, 1024, kernel_size=4, stride=1, padding=0, output_padding=0),
        #                 norm_layer(1024), activation]
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model_2 += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                    norm_layer(int(ngf * mult / 2)), activation]
            # if n_downsampling - i < 5:
            #     mult = 2**(n_downsampling - i)
            #     model_2 += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
            #             norm_layer(int(ngf * mult / 2)), activation]
            # else:
            #     model_2 += [nn.ConvTranspose2d(1024, 1024, kernel_size=3, stride=2, padding=1, output_padding=1),
            #             norm_layer(1024), activation]
        # model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]        
        self.model = nn.Sequential(*model)
        self.model_2 = nn.Sequential(*model_2)

        self.merge = nn.Sequential(nn.Conv2d(512+256, 512, kernel_size=3, stride=1, padding=1),
                    norm_layer(512), activation)

        ### resnet blocks
        # BN = []
        # latent_size = 512
        # for i in range(n_blocks):
        #     BN += [
        #         ResnetBlock_Adain(1024, latent_size=latent_size, padding_type=padding_type, activation=activation)]
        # self.BottleNeck = nn.Sequential(*BN)

    def forward(self, input, dlatents=None, middle_feature=None):
        # return self.model(input)
        if middle_feature is None:
            x = self.model(input)
        else:
            # x = self.model(input)
            x = torch.cat([self.model(input), middle_feature], dim=1)
            x = self.merge(x)
        # if dlatents is not None:
        #     for i in range(len(self.BottleNeck)):
        #         x = self.BottleNeck[i](x, dlatents)
        # x = self.con_former(x, middle_feature)
        return self.model_2(x), x

class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d, 
                 padding_type='reflect', opt=None):
        assert(n_blocks >= 0)
        super(GlobalGenerator, self).__init__()        
        activation = nn.ReLU(True)        
        
        ### resnet blocks
        model_2 = list()
        
        mult = 2**n_downsampling
        
        if opt.times:
            # model = [nn.Conv2d(32, int(ngf * mult/2), kernel_size=3, stride=1, padding=1),
            #         norm_layer(int(ngf * mult/2)), activation]
            model = [nn.Conv2d(32 * opt.clip_length, int(ngf * mult/2), kernel_size=3, stride=1, padding=1),
                    norm_layer(int(ngf * mult/2)), activation]
            # self.model = nn.ModuleList([
            #                 nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            #                     norm_layer(32), activation),
            #                 nn.Sequential(nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            #                     norm_layer(32), activation),
            #                 nn.Sequential(nn.Conv2d(32, 32, kernel_size=7, stride=1, padding=3),
            #                     norm_layer(32), activation)])
        else:
            model = [nn.Conv2d(32, ngf * mult, kernel_size=3, stride=1, padding=1),
                        norm_layer(ngf * mult), activation]

        for i in range(n_blocks):
            # model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
            model_2 += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        
        ### upsample
        # model_2 += [nn.ConvTranspose2d(1024, 1024, kernel_size=4, stride=1, padding=0, output_padding=0),
        #                 norm_layer(1024), activation]
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model_2 += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                    norm_layer(int(ngf * mult / 2)), activation]
                
        self.model = nn.Sequential(*model)
        self.model_2 = nn.Sequential(*model_2)

        self.merge = nn.Sequential(nn.Conv2d(512+256, 512, kernel_size=3, stride=1, padding=1),
                    norm_layer(512), activation)

    def forward(self, input, middle_feature=None):
        # return self.model(input)
        if middle_feature is None:
            x = input
            # x = self.model(input)
        
        else:
            x = input
            
            # middle_feature_list = [middle_feature]
            # for model in self.model:
            #     middle_feature_list.append(model(middle_feature))
            # x = torch.cat([input, *middle_feature_list], dim=1)
            
            middle_feature = self.model(middle_feature)
            x = torch.cat([input, middle_feature], dim=1)
            # x = self.merge(x)
        
        return self.model_2(x)

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class Encoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsampling=4, norm_layer=nn.BatchNorm2d):
        super(Encoder, self).__init__()        
        self.output_nc = output_nc        

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), 
                 norm_layer(ngf), nn.ReLU(True)]             
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), nn.ReLU(True)]

        ### upsample         
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]        

        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model) 

    def forward(self, input, inst):
        outputs = self.model(input)

        # instance-wise average pooling
        outputs_mean = outputs.clone()
        inst_list = np.unique(inst.cpu().numpy().astype(int))        
        for i in inst_list:
            for b in range(input.size()[0]):
                indices = (inst[b:b+1] == int(i)).nonzero() # n x 4            
                for j in range(self.output_nc):
                    output_ins = outputs[indices[:,0] + b, indices[:,1] + j, indices[:,2], indices[:,3]]                    
                    mean_feat = torch.mean(output_ins).expand_as(output_ins)                                        
                    outputs_mean[indices[:,0] + b, indices[:,1] + j, indices[:,2], indices[:,3]] = mean_feat                       
        return outputs_mean


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, 
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        # (30,32,3,instance,false,3,false)
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
     
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:                                
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))                                   
            else:
                setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):        
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)
        return result


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)        


from torchvision import models
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
