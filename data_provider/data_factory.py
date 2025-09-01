from data_provider.data_loader import Dataset_Custom
from torch.utils.data import DataLoader

def data_provider(args, flag):
    """
    Unified data provider for all tasks using CustomDataset.
    Supports training, validation, testing, classification, and anomaly detection.
    """

    # Decide if time features are encoded
    timeenc = 0 if args.embed != 'timeF' else 1

    # Shuffle only for training
    shuffle_flag = flag.lower() == 'train'
    drop_last = False
    batch_size = args.batch_size
    freq = args.freq

    # Dataset initialization
    data_set = Dataset_Custom(
        args=args,
        root_path=args.root_path,
        data_path=getattr(args, "data_path", None),
        flag=flag,
        size=[args.seq_len, getattr(args, "label_len", 0), getattr(args, "pred_len", 0)],
        features=getattr(args, "features", "S"),
        target=getattr(args, "target", "OT"),
        timeenc=timeenc,
        freq=freq,
        seasonal_patterns=getattr(args, "seasonal_patterns", None),
        augmentation_ratio=getattr(args, "augmentation_ratio", 0.0)
    )

    # Collate function only needed for classification tasks
    collate = None
    if args.task_name == 'classification':
        from data_provider.uea import collate_fn
        collate = lambda x: collate_fn(x, max_len=args.seq_len)

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        collate_fn=collate
    )

    print(f"{flag} set size: {len(data_set)}")
    return data_set, data_loader
