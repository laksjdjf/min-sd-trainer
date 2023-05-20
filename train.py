from PIL import Image
import os
import sys
import glob
from tqdm import tqdm
import importlib
from omegaconf import OmegaConf
import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer

def get_attr_from_config(config_text: str):
    module = ".".join(config_text.split(".")[:-1])
    attr = config_text.split(".")[-1]
    return getattr(importlib.import_module(module), attr)

class SimpleDataset(Dataset):
    def __init__(self, path, size):
        self.file_list = []
        [self.file_list.extend(glob.glob(f'{path}' + '/*.' + e)) for e in ['jpg', 'jpeg', 'png', 'bmp', 'webp']]
        self.transform = transforms.Compose(
            [
                transforms.Resize(size),
                transforms.ToTensor(), 
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )
    def __len__(self):
        return len(self.file_list)
    def __getitem__(self, i):
        img = Image.open(self.file_list[i])
        with open(os.path.splitext(self.file_list[i])[0] + ".txt" ,"r") as f:
            caption = f.read()
        return {"image":self.transform(img),"caption":caption}

def main(config):
    os.makedirs(config.output,exist_ok=True)
    os.makedirs(config.image_log,exist_ok=True)

    size = config.resolution.split(",") 
    size = (int(size[0]),int(size[-1])) # width,height
    
    device = torch.device('cuda')
    weight_dtype = torch.bfloat16 if config.amp == "bfloat16" else torch.float16 if config.amp else torch.float32
    
    tokenizer = CLIPTokenizer.from_pretrained(config.model, subfolder='tokenizer')
    
    text_encoder = CLIPTextModel.from_pretrained(config.model, subfolder='text_encoder')
    text_encoder.requires_grad_(False)
    text_encoder.train(False)
    
    vae = AutoencoderKL.from_pretrained(config.model, subfolder='vae')
    vae.requires_grad_(False)
    vae.eval()

    unet = UNet2DConditionModel.from_pretrained(config.model, subfolder='unet')
    unet.set_use_memory_efficient_attention_xformers(config.use_xformers)
    unet.requires_grad_(True)
    unet.train()

    if config.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    
    optimizer_cls = get_attr_from_config(config.optimizer) # configからoptimzerを取得
    optimizer = optimizer_cls(unet.parameters(),lr=config.lr)
    
    text_encoder.to(device,dtype=weight_dtype)
    vae.to(device,dtype=weight_dtype)
    unet.to(device,dtype=torch.float32) #学習対称はfloat32
    
    noise_scheduler = DDPMScheduler.from_pretrained(config.model,subfolder='scheduler')
    
    dataset = SimpleDataset(config.dataset,size)
    dataloader = DataLoader(dataset,batch_size=config.batch_size,num_workers=0,shuffle=True)

    scaler = torch.cuda.amp.GradScaler(enabled=config.amp != False) #AMP用のスケーラー
    
    progress_bar = tqdm(range((config.epochs) * len(dataloader)), desc="Total Steps", leave=False)
    loss_ema = None #損失の指数移動平均（監視用）
    
    for epoch in range(config.epochs):
        for batch in dataloader:

            tokens = tokenizer(batch["caption"], max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors='pt').input_ids.to(device)
            encoder_hidden_states = text_encoder(tokens, output_hidden_states=True).last_hidden_state.to(device)
            
            latents = vae.encode(batch['image'].to(device, dtype=weight_dtype)).latent_dist.sample().to(device) * 0.18215
            
            noise = torch.randn_like(latents)
            bsz = latents.shape[0] # バッチサイズ
            
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps) # stepsに応じてノイズを付与する
            
            with torch.autocast("cuda",enabled=config.amp != False):
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample # 付与したノイズを推定する
                
            if config.v_prediction:
                noise = noise_scheduler.get_velocity(latents, noise, timesteps) # v_predictionではvelocityを予測する。

            loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float(), reduction="mean") # 本物ノイズと推定ノイズの誤差
            
            scaler.scale(loss).backward() #AMPの場合アンダーフローを防ぐために自動でスケーリングしてくれるらしい。
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            loss_ema = loss.item() if loss_ema is None else loss_ema * 0.9 + loss.item() * 0.1
            logs={"loss":loss_ema}
            progress_bar.update(1)
            progress_bar.set_postfix(logs)
        
        if epoch % config.save_n_epochs == config.save_n_epochs - 1: #モデルのセーブと検証画像生成
            pipeline = StableDiffusionPipeline.from_pretrained(
                config.model,text_encoder=text_encoder,vae=vae,unet=unet,tokenizer=tokenizer,
                feature_extractor = None,safety_checker = None
            )
            with torch.autocast('cuda', enabled=config.amp != False):    
                image = pipeline(batch["caption"][0],width=size[0],height=size[1]).images[0]
            image.save(os.path.join(config.image_log,f'image_log_{str(epoch).zfill(3)}.png'))
            pipeline.save_pretrained(f'{config.output}')
            del pipeline
        
if __name__ == "__main__":
    config = OmegaConf.load(sys.argv[1])
    main(config)