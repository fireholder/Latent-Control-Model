import torch
from lcm import LatentControlModel
from arg_test import get_args
import numpy as np
import torch.optim as optim
import ipdb
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

args = get_args()

device = torch.device("cuda:0" if args.cuda else "cpu")
writer = SummaryWriter('logs')

if args.cuda and torch.cuda.is_available():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

model = LatentControlModel(
                num_grid = 4,
                num_stack = 1,
                action_space_n = 1,
                obs_size = 84,
                random_noise_frame = True,
                epsilon = 1,
                ob_bound = 0,
                model_structure = args.model_structure['LatentControlModel'],
                is_action_conditional = False,
            )

model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.00001, betas=(0.0,0.9))

sampled = {}

x = np.zeros((84,84))
x1 = x.copy()
x1[:42,:42] = 1
x2 = x.copy()
x2[:42,42:] = 1
x3 = x.copy()
x3[:42,:] = 1
sampled['states'] = torch.Tensor(np.array([x,x1,x2,x3]))
sampled['next_states'] = torch.Tensor(np.array([x,x2,x,x2]))


# sampled['states'] = torch.Tensor(np.array([[[0,0]],[[1,0]],[[0,1]],[[1,1]]]))
# sampled['states'] = torch.Tensor(np.array([[[0,0,0,1],[0,0,0,0],[0,0,0,0],[0,0,0,0]],[[0,1,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]]))
sampled['states'] = torch.unsqueeze(sampled['states'],1)
sampled['states'] = Variable(sampled['states'], requires_grad=True).to(device)
# sampled['next_states'] = torch.Tensor(np.array([[[0.1,0.1]],[[0.1,0.9]],[[0.1,0.1]],[[0.1,0.9]]]))
# sampled['next_states'] = torch.Tensor(np.array([[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],[[0,0,0,1],[0,0,0,0],[0,0,0,0],[0,0,0,0]]]))
sampled['next_states'] = torch.unsqueeze(sampled['next_states'],1)
sampled['next_states'] = Variable(sampled['next_states'], requires_grad=True).to(device)
sampled['actions'] = torch.Tensor(np.array([[0],[0],[0],[0]]))
# sampled['actions'] = torch.Tensor(np.array([[0],[0]]))
sampled['actions'] = Variable(sampled['actions'], requires_grad=True).to(device)
# sampled['states'] = sampled['states'].repeat(10,1,1,1)
# sampled['next_states'] = sampled['next_states'].repeat(10,1,1,1)
# sampled['actions'] = sampled['actions'].repeat(10,1)
epoch = 0

while True:
    # if epoch > 1000:
    #     ipdb.set_trace()
    optimizer.zero_grad()
    model.train()
    loss_transition, loss_transition_each, loss_ent_latent = model(
                    last_states    = sampled['states'],
                    now_states     = sampled['next_states'],
                    onehot_actions = sampled['actions'],
                )
    loss_transition = loss_transition.mean(dim=0,keepdim=False)
    '''integrate losses'''
    loss = loss_transition + loss_transition_each + 0.001*loss_ent_latent
    # if epoch > 50:
    #     ipdb.set_trace()
    '''backward'''
    writer.add_scalar('Train/Loss',loss.item(), epoch)
    writer.flush()
    loss.backward(retain_graph=True)
    # ipdb.set_trace()
    '''optimize'''
    optimizer.step()
    epoch += 1


