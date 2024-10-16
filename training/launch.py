from model import IRT
from data_loader import load_data, prepare_loaders
from train import train
from utils import adaptest_save

config = {
    'learning_rate': 0.0002,
    'batch_size': 2048,
    'num_epochs': 1000,
    'num_dim': 1, 
    'device': 'cpu',
}

dataset = 'math'
triplets, metadata = load_data(dataset)
train_loader, val_loader = prepare_loaders(triplets, config['batch_size'])

snum = metadata['num_students']
qnum = metadata['num_questions']

model = IRT(snum, qnum, config['num_dim'])

results = train(train_loader, val_loader, model, config)
adaptest_save(model, '../scripts/model/' + dataset + '/beta_params.pth')
