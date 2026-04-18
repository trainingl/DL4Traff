import sys
sys.path.append('../../')

from datetime import datetime
from utils.utils import *
from utils.data_loader import *
from STID_Config import args
from STID_Trainer import Trainer
from models.STID import STID as Network


def load_data(args):
    data_loader = load_dataset_v2(args.dataset_dir, args.batch_size, args.batch_size, args.batch_size)
    scaler = data_loader['scaler']
    return data_loader, scaler


def get_log_dir(model, dataset):
    current_time = datetime.now().strftime('%Y%m%d%H%M%S')
    current_dir = os.path.abspath(os.path.join(os.getcwd(), "../../"))
    log_dir = os.path.join(current_dir, 'logs', model, dataset, current_time)
    return log_dir


def generate_model_components(args):
    # 1. model
    model = Network(
        num_nodes=args.num_nodes, 
        input_dim=args.input_dim, 
        embed_dim=args.embed_dim, 
        input_len=args.seq_len, 
        output_len=args.horizon, 
        num_layer=args.num_layer,
        node_dim=args.node_dim, 
        temp_dim_tod=args.temp_dim_tod, 
        temp_dim_dow=args.temp_dim_dow, 
        time_of_day_size=args.time_of_day_size, 
        day_of_week_size=args.day_of_week_size,
        if_time_of_day=args.if_time_of_day, 
        if_day_of_week=args.if_day_of_week, 
        if_spatial=args.if_spatial
    )
    model = model.to(args.device)
    print_model_parameters(model, only_num=False)
    # 2. loss function
    if args.loss_func == 'masked_mae':
        loss = MaskedMAELoss()
    elif args.loss_func == 'mae':
        loss = torch.nn.L1Loss().to(args.device)
    elif args.loss_func == 'mse':
        loss = torch.nn.MSELoss().to(args.device)
    elif args.loss_func == 'smoothloss':
        loss = torch.nn.SmoothL1Loss().to(args.device)
    elif args.loss_func == 'huberloss':
        loss = torch.nn.HuberLoss().to(args.device)
    else:
        raise ValueError
    # 3. optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=1e-8)
    # 4. learning rate decay
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=args.milestones,
        gamma=args.gamma,
        verbose=False
    )
    return model, loss, optimizer, lr_scheduler


def init_seed(seed):
    """
    Disable cudnn to maximize reproducibility
    """
    torch.cuda.cudnn_enabled = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == '__main__':
    init_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.device[5]))
    else:
        args.device = 'cpu'

    data_loader, scaler = load_data(args)
    args.log_dir = get_log_dir(args.model, args.dataset)
    model, loss, optimizer, lr_scheduler = generate_model_components(args)

    trainer = Trainer(
        args=args,
        data_loader=data_loader,
        scaler=scaler,
        model=model,
        loss=loss,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler
    )
    if args.mode == 'train':
        trainer.train()
    elif args.mode == 'test':
        checkpoint = "../../log/STID/PMBS08/20240405124036/PMBS08_STID_best_model.pth"
        trainer.test(args, model, data_loader, scaler, trainer.logger, save_path=checkpoint)
    else:
        raise ValueError