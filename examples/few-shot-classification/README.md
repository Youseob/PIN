# Prompted Few-Shot Classification Example

The script below runs a 16-shot classification experiment, with options for `task_lm` and `dataset`. For each dataset, we provide 5 different 16-shot training sets, toggled by `dataset_seed`.

To run PIN,
```
python run_fsc.py dataset=[sst-2, yelp-2, mr, cr, agnews, sst-5, yelp-5] \
                  policy_lm=facebook/opt-125m \
                  prompt_train_batch_size=16 \
                  adaptor_model=true \
                  project_name=fsc \
                  off_train_batch_size=256 \
                  prompt_length=5 \
                  max_decoding_length=5 \
                  algo=sql \
                  training_mode=sql-offpolicy \
                  logit_bias=false \
                  loss_impl=v1 \
                  fluent=true \ 
                  fluent_top_k= 10000 \
                  random_seed=[any integer (optional)]
```

You can find additional hyperparameters in `fsc_config.yaml` and the default configs imported by `run_fsc.py`

## Evaluation

After you train a prompt, you can evaluate it on a given dataset with the following commands
```
cd evaluation
python run_eval.py \
    dataset=[sst-2, yelp-2, mr, cr, agnews, sst-5, yelp-5] \
    task_lm=[distilroberta-base, roberta-base, roberta-large, \
             distilgpt2, gpt2, gpt2-medium, gpt2-large, gpt2-xl] \
    prompt=[any prompt in string form, e.g. "Absolutely", \
    and for a special case of leading whitespace prompt, \
    we have to use "prompt=\" Absolutely\"" instead]
```


