from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from trainer_speechLLM_llama_v3_1b_noquant import SpeechLLM
from dataset_biasingLLM_llama_v3 import InstructionalBiasingASRDataset, MyCollator_biasingLLM
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from torch.utils.data import Dataset, DataLoader
import wandb
#../qwen2/0.5B/
#../llama2_hf/7B-chat/
if __name__ == "__main__":
    log_path = 'libri100_speechLLM_llama1b'

    model_config = {
                'audio_encoder_name': "whisper_large",
                'audio_dim': 1280,
                'llm_dim': 2048, # Llama-8b: 4096
                'llm_mode': 'train',
                'hidden_dim': 1280,
                'use_quantity_loss' : True,  # 是否使用 quantity loss
                'lambda2' : 1.0,  # quantity loss 權重
                'gop_encoder_name' : "conv-branchformer",
                'biasing_module_name': "transformer",
                'biasing_heads': 4,
                'biasing_depths': 2,
                'connector_name': 'cnn', # for audio feats
                'connector_k': 4,
                'llm_name': "../llama3.2/huggingface/Llama-3.2-1B",
                'finetune_encoder': False,
                'use_lora': True,
                'lora_r': 8, # 8->20M
                'lora_alpha': 32,
                'max_lr': 3e-5, #3e-5
                'total_training_step': 300000,
                'warmup_steps': 100,
                'train_batch_per_epoch': 20000,
                'grad_accumulate_steps': 25
        }  
     
    wandb.login(key='1426f4f3ee913e566af445200fca103b3607cd03')
    logger = WandbLogger(
        project="libri100_speechLLM_llama1b",
        name=log_path,
        log_model=True,  # 記錄模型檢查點
        config=model_config  # 自動記錄所有配置參數
    )
    
    model = SpeechLLM(**model_config)
    tokenizer = model.llm_tokenizer
    audio_proc = model.audio_proc
    my_collator = MyCollator_biasingLLM(audio_proc, tokenizer)

    train_dataset = InstructionalBiasingASRDataset(
        # json_file='data/LibriSpeech/biasingASR_json/train_clean_100_full.json',
        json_file='./data/LibriSpeech/train_clean_100_full.json',
        mode='train')

    val_dataset = InstructionalBiasingASRDataset(
        # json_file='data/LibriSpeech/biasingASR_json/dev_clean_full.json',
        json_file='./data/LibriSpeech/dev_clean_full.json',
        mode='train')

    print(len(train_dataset), len(val_dataset))
    print(tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        collate_fn=my_collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=my_collator,
    )

    checkpoint_callback = ModelCheckpoint(dirpath=f"./exp4/speechLLM_llama1b_CIFwithQuantityLoss_prompt", filename=log_path+'/{epoch}', save_top_k=1, monitor="val/loss", save_last=True)
    early_stop_callback = EarlyStopping(monitor="val/loss", min_delta=0.00, patience=3, verbose=True, mode="min")
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = Trainer(
            max_epochs=model_config['total_training_step']//model_config['train_batch_per_epoch'],
            devices=1, 
            accelerator="auto",
            strategy=DDPStrategy(find_unused_parameters=True),
            limit_train_batches=model_config['train_batch_per_epoch'], 
            limit_val_batches=model_config['train_batch_per_epoch'], 
            log_every_n_steps=model_config['train_batch_per_epoch'], 
            enable_checkpointing=True, 
            callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
            fast_dev_run=False, 
            logger=logger,  # 使用 WandbLogger
            accumulate_grad_batches=model_config['grad_accumulate_steps'],
    )

    # Train from checkpoint
    # trainer.fit(model, train_loader, val_loader, ckpt_path='./exp4/speechLLM_llama1b_CIFwithQuantityLoss_prompt/libri100_speechLLM_llama1b/epoch=0.ckpt')

    # Train from scratch
    trainer.fit(model, train_loader, val_loader)

