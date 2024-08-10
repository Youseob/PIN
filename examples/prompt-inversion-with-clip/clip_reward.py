import torch
from torch.utils.data import Dataset
import numpy as np
import itertools
from typing import List, Tuple, Union, Dict, Any, Optional
from transformers import pipeline, AutoTokenizer
# from bert_score import BERTScorer
from collections import defaultdict
from PIL import Image
from io import BytesIO
import requests

from rlprompt.rewards import BaseReward
# from rlprompt.models.base_model import SUPPORTED_LMS
import open_clip

coco_url_list = ["http://images.cocodataset.org/train2017/000000125547.jpg","http://images.cocodataset.org/train2017/000000037527.jpg",
"http://images.cocodataset.org/train2017/000000552584.jpg","http://images.cocodataset.org/train2017/000000490129.jpg",
"http://images.cocodataset.org/train2017/000000548337.jpg","http://images.cocodataset.org/train2017/000000458160.jpg",
"http://images.cocodataset.org/train2017/000000212733.jpg","http://images.cocodataset.org/train2017/000000035599.jpg",
"http://images.cocodataset.org/train2017/000000067548.jpg","http://images.cocodataset.org/train2017/000000147961.jpg",
"http://images.cocodataset.org/train2017/000000012764.jpg","http://images.cocodataset.org/train2017/000000507721.jpg",
"http://images.cocodataset.org/train2017/000000335304.jpg","http://images.cocodataset.org/train2017/000000257928.jpg",
"http://images.cocodataset.org/train2017/000000244368.jpg","http://images.cocodataset.org/train2017/000000178423.jpg",
"http://images.cocodataset.org/train2017/000000370055.jpg","http://images.cocodataset.org/train2017/000000096048.jpg",
"http://images.cocodataset.org/train2017/000000233477.jpg","http://images.cocodataset.org/train2017/000000120199.jpg"]

laion_url_list = [
    "https://i1.wp.com/musicexistence.s3.amazonaws.com/wp-content/uploads/2015/08/Whitesnake-716-of-38.jpg?w=422&h=282&ssl=1",
    "http://img1.imagesbn.com/p/9780439872508_p0_v1_s260x420.jpg",
    "https://i1.wp.com/withtwospoons.com/wp-content/uploads/2017/05/Asian-Chicken-Salad-with-Grilled-Vegetables-2.jpg?resize=1024%2C683&",
    "https://photos.smugmug.com/Daily-Local/Downingtown-East-graduation/i-X7pKKD2/0/0333fe86/Ti/DLN-L-DtownEast-0609-8-Ti.jpg",
    "https://cdn.shopify.com/s/files/1/0081/1260/3217/products/9fca2fb779454048ac98c5121c2c5d53.jpg?v=1568909657",
    "http://ecx.images-amazon.com/images/I/51YheCPWWSL._SL160_.jpg",
    "http://photos.listhub.net/MRIS/MC8061358/0",
    "https://dwu32cgxelq1c.cloudfront.net/local_newspapers/sites/2/2016/06/13096140_1018110374_40783-270x203.jpg",
    "https://a0.cdn.japantravel.com/photo/17914-103974/360x240!/chiba-chiba-zoological-park-103974.jpg",
    "https://d7hftxdivxxvm.cloudfront.net/?resize_to=fit&width=400&height=305&quality=80&src=https%3A%2F%2Fd32dm0rphc51dk.cloudfront.net%2FRyskG8M23ca56AMxsCbFKA%2Flarge.jpg",
    "https://i.pinimg.com/736x/e0/e0/3b/e0e03b79df414b8b5698a5c239ae96b4.jpg",
    "https://d31l02nbp0owar.cloudfront.net/m/t/125/1246048/a-0081.jpg",
    "https://photos.smugmug.com/Travel/Paris/i-LWnPWhD/1/S/197810 La Marseillaise- Arc de Triomphe- Paris-S.jpg",
    "https://www.totalsportspicks.com/wp-content/uploads/2020/03/ja-morant-of-the-Grizzlies.jpg",
    "https://cdn.shopify.com/s/files/1/0003/5615/5445/products/MEG-0055-OV-S-BL-DI-W-R__4_720x720.jpg?v=1533984183",
    "https://images.classmates.com/yearbooks/7/7/a/0/77a046dd8919dfe503a4d8bccac035ac/155/0020.jpg",
    "https://static.timesofisrael.com/njjewishnews/uploads/2018/12/MoroccoHassanIIMosquePMB-640x400.jpg",
    "http://images.shopflowers.net/images/products/HW0_372952.jpg",
    "https://gori.me/uploads/2015/03/ipad-air-igzo-display-723x603.png",
    "https://fullopportunities.com/wp-content/uploads/2018/11/Optimized-Optimized-HEC-Scholarship-2019-study-in-korea-696x458.jpg"
]

def download_image(url):
    try:
        response = requests.get(url)
    except:
        return None
    return Image.open(BytesIO(response.content)).convert("RGB")

class ClipReward(BaseReward):
    def __init__(
        self,
        clip_model: str, # name clip model, "ViT-H-14",
        clip_pretrain: str, # "laion2b_s32b_b79k"
        target_image_idx: Optional[int]=-1,
        dataset_name: Optional[str]="celeba",
        # style_batch_size: int,
    ):
        self.reward_device = 0  # TODO
        self.task_lm = 'clip'
        # assert task_lm in SUPPORTED_LMS
        print('Task LM:', clip_model)
        self.tokenizer = open_clip.get_tokenizer(clip_model)
        self.clip_model, _ , preprocess = open_clip.create_model_and_transforms(
            clip_model, pretrained=clip_pretrain, device=self.reward_device)

        self._counter = 0
        self.tokens_explored = set()
        if target_image_idx==-1:
            urls = ["https://a.1stdibscdn.com/alexander-averin-paintings-pony-riding-on-the-beach-post-impressionist-style-oil-painting-for-sale-picture-6/a_7443/a_28523631526593507117/Promenade_detalle_5_master.JPG?disable=upscale&auto=webp&quality=60&width=1318",]
            target_images = list(filter(None,[download_image(url) for url in urls]))
        elif dataset_name=="coco":
            urls = [coco_url_list[target_image_idx],]
            target_images = list(filter(None,[download_image(url) for url in urls]))
        elif dataset_name=="laion":
            urls = [laion_url_list[target_image_idx]]
            target_images = list(filter(None,[download_image(url) for url in urls]))
        else:
            target_images = [Image.open("./target_imgs/"+dataset_name+"/"+str(target_image_idx)+".jpg"),]
        curr_images = [preprocess(i).unsqueeze(0) for i in target_images]
        curr_images = torch.concatenate(curr_images).to(self.reward_device)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(curr_images)
        self.norm_image_features  = image_features / image_features.norm(dim=1, keepdim=True)
        
        self.max_reward = -1.
        self.max_prompt = "None"
    
    @torch.no_grad()
    def forward(
        self,
        # source_texts: List[str],
        # target_labels: List[str],
        prompt_strs: List[str],
        to_tensor: bool,
        mode: str
    ) -> Tuple[Union[List[float], torch.Tensor], Dict[str, Any]]:
        if mode == 'train':
            self._counter += 1
        elif mode == "infer":
            pass
        else:
            raise ValueError


        quantities_to_log: Dict[str, torch.Tensor] = defaultdict()
        input_tokens = self.tokenizer(prompt_strs).to(self.reward_device) # (batch_size, seq_len)
        text_features = self.clip_model.encode_text(input_tokens)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        
        rewards_tensor = self.norm_image_features @ text_features.t() # (1, batch_size_text)
        rewards_tensor = rewards_tensor.squeeze(0)
        top_index = torch.argmax(rewards_tensor, dim=-1)
        
        if self._counter % 50 == 0 :
            print('-'*20)
            print(f"{self._counter} | {prompt_strs[top_index]} |")
            print(f"rewards: {round(rewards_tensor[top_index].cpu().item(), 2)}")
        # self.tokens_explored = \
        #     self.tokens_explored.union(*[set(p) for p in prompt_tokens])
        # quantities_to_log["num_tokens_explored"].append(
        #     torch.tensor(len(self.tokens_explored)).float())
        if self.max_reward < rewards_tensor[top_index].cpu().item():
            self.max_prompt = prompt_strs[top_index]
            self.max_reward = rewards_tensor[top_index].cpu().item()
        
        rewards_log = {
            "max_explored_reward": rewards_tensor[top_index]
        }        
        if to_tensor is True:
            return rewards_tensor, rewards_log
        else:
            return rewards_tensor.tolist(), rewards_log

if __name__=="__main__":
    reward = ClipReward('ViT-H-14', 'laion2b_s32b_b79k')
    output_tokens = ['HiHelloggggg', 'ABC']
    output = reward(output_tokens, to_tensor=True, mode='train')