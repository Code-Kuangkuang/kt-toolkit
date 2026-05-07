from scripts.train import launch_train

launch_train(
    dataset_name="assist2009",
    model_name="gbktv2",
    fold=0,
    num_epochs=200,
    use_wandb=0,
    save_dir="saved_model",
    seed=3407,
 )