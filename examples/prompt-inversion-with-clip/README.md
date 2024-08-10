# Textual inversion from images

The script below runs textual inversion from images experiment, with options for `dataset`. For each dataset, we provide 10 target images.

To run PIN,
```
python run_pic.py dataset_name=[coco, lexica, laion] \
                  target_image_idx=[0~9] \
                  policy_lm=facebook/opt-350m \
                  prompt_train_batch_size=32 \
                  adaptor_model=true \
                  project_name=pic \
                  off_train_batch_size=256 \
                  prompt_length=8 \
                  max_decoding_length=8 \
                  algo=sparse-ql \
                  training_mode=sparse-ql-offpolicy \
                  logit_bias=false \ 
                  loss_impl=v1 \
                  fluent=true \
                  fluent_top_k=10000 \
                  reward_shaping_new_max=1 \
                  max_train_steps=20000 \
                  random_seed=0
```

You can find additional hyperparameters in `pic_config.yaml` and the default configs imported by `run_pic.py`

## Evaluation

After you train a prompt, you can evaluate it on a given dataset with the 'eval.py' 



