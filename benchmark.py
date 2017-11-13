import time
import models
import torch
from torch.autograd import Variable

model = models.VGG_DECONV(13)
model.copy_params_from_vgg16()

model.eval()
name = 'hung-net'

for i in range(50):
    input = torch.rand(1, 3, 480, 640)
    input = Variable(input, volatile=True)
    t2 = time.time()
    out = model(input)
    t3 = time.time()
    print(out.size())
    print('%10s : %f' % (name, t3 - t2))
