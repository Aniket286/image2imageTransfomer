import yaml
from trainer import Trainer
from torch.autograd import Variable
import torchvision.utils as vutils
import sys
import torch
import os
import numpy as np
from torchvision import transforms
from PIL import Image

checkpoint = 'models/new_edgetohandbag.npy'
trainer = 'MUNIT' 
output_path = '.'
output_folder='outputs'
input='inputs/edge.jpg'
config_path = 'configs/demo_edges2handbags_folder.yaml'
style=''
num_style=10

with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)

config['vgg_model_path'] = output_path
style_dim = config['gen']['style_dim']
trainer = Trainer(config)

state_dict = np.load(checkpoint,allow_pickle='TRUE').item()
trainer.gen_a.load_state_dict(state_dict['a'])
trainer.gen_b.load_state_dict(state_dict['b'])
    
trainer.cuda()
trainer.eval()
encode = trainer.gen_a.encode # encode function
style_encode = trainer.gen_b.encode  # encode function
decode = trainer.gen_b.decode  # decode function

with torch.no_grad():
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    image = Variable(transform(Image.open(input).convert('RGB')).unsqueeze(0).cuda())
    style_image = Variable(transform(Image.open(style).convert('RGB')).unsqueeze(0).cuda()) if style != '' else None

    # Start testing
    content, _ = encode(image)
    
style = Variable(torch.randn(num_style, style_dim, 1, 1).cuda())
for j in range(10):
    s = style[j].unsqueeze(0)
    outputs = decode(content, s)
    outputs = (outputs + 1) / 2.
    path = os.path.join(output_folder, 'output{:03d}.jpg'.format(j))
    vutils.save_image(outputs.data, path, padding=0, normalize=True)