from settings import Settings, CocoDataset, DotEnvControl
from train import StyleTrasferTrainer
from tg_sender import TgSender

def main():
    params = Settings().parse_args()
    print(params)
    path='../config/.env_local'
    env_control = DotEnvControl(path)

    tg_sender = None
    if params.result_to_tg != 0:
        tg_sender = TgSender(env_control.TG_NOTIFY_BOT_TOKEN)        

    if CocoDataset(env_control.ROOT_SRC_DIR, env_control.TRAIN_DATASET) \
        .is_dataset_exist:
        fit = StyleTrasferTrainer(params, env_control, tg_sender)
        fit.train()


if __name__ == "__main__":
    main()