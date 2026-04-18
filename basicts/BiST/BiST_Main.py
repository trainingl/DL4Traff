import sys
sys.path.append('../../')

from datetime import datetime
from utils.utils import *
from utils.data_loader import *
from BiST_Config import args
from BiST_Trainer import Trainer
from models.BiST import BiST as Network


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
        model_dim=args.model_dim, 
        prompt_dim=args.prompt_dim, 
        num_layers=args.num_layers, 
        input_len=args.seq_len, 
        output_len=args.horizon, 
        time_of_day_size=args.time_of_day_size, 
        day_of_week_size=args.day_of_week_size, 
        kernel_size=args.kernel_size, 
        hidden_dim=args.hidden_dim, 
        num_cores=args.num_cores,
        extra_type=args.extra_type,
        same=args.same,
        rp_layer=args.rp_layer, 
        datadriven_adj=args.datadriven_adj, 
        datadriven_adj_dim=args.datadriven_adj_dim, 
        adaptive_adj=args.adaptive_adj, 
        adaptive_adj_dim=args.adaptive_adj_dim, 
        mrf=args.mrf
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
        checkpoint = "../../logs/BiST/PEMS08/20260403215653/PEMS08_BiST_best_model.pth"
        trainer.test(args, model, data_loader, scaler, trainer.logger, save_path=checkpoint)
    else:
        raise ValueError