# Prompted Text Style Transfer Example

## Setup

In addition to the dependencies for RLPrompt, you can install the additional dependencies for text style transfer with
```
pip install -r requirements.txt
```

## Training Prompts for Style Transfer

You can download the pretrained style classifiers for training by running the script below
```
python scripts/download_tst_classifiers.py \
    --model_name [yelp-train,
                  shakespeare-train-100-0,
                  shakespeare-train-100-1,
                  shakespeare-train-100-2]
```

The `100` and `0,1,2` for `shakespeare` refer to the training data size and the random seed we used to sample the training data, respectively. 


After that, you can run the experiment with the command below. Below are the explanations for some important parameters: 
- `dataset_seed`: The random seed of Shakespeare training sets. Skip this for Yelp
- `direction`: `0_to_1` refers to negative-to-positive for Yelp and old-to-modern for Shakespeare, while `1_to_0` means the opposite
- `lower_outputs`: Whether to manually set outputs to lower case. Due to dataset properties, we set it to `true` for Yelp and `false` for Shakespeare

You can find additional hyperparameters in `tst_config.yaml` and the default configs imported by `run_tst.py`.

To run PIN, 
```
python run_tst.py 
    dataset=yelp \
    direction=[0_to_1, 1_to_0] \
    style_classifier=[style_classifier path, ex.) /style_classifiers/yelp-bert-base-uncased-train/] \
    task_lm=facebook/opt-1.3b \
    policy_lm=facebook/opt-125m \ 
    off_train_batch_size=256 \
    adaptor_model=true \
    project_name=yelp \
    prompt_train_batch_size=8 \
    algo=sparse-ql \
    training_mode=sparse-ql-offpolicy \
    logit_bias=false \
    loss_impl=v1 \
    fluent=true \
    fluent_top_k=10000  
```


## Using Your Own Data

To run text style transfer using your own data, you would need to provide the following:
1. A text classifier which defines the style
2. A function for reading the data 

### Style Classifier
Our code can technically work with any binary text classifier using the `transformers` library. For out-of-the-box models, [Huggingface Model Hub](https://huggingface.co/models?pipeline_tag=text-classification&sort=downloads) is a great place to start with. 
If you'd like to train your own model, [this tutorial](https://huggingface.co/docs/transformers/training) provides an excellent example for fine-tuning a pretrained LM to serve as classifier. 

Once you've decided on the style classifier, you may add its name (in case of Model Hub models) or path (in case of local models) to `style_classifier_dict` under `tst_helpers.py`, which allows it to be called by its `dataset` name. 

### Data Loading Function
We load our data using the `load_text_style_transfer_dataset()` function in the `tst_data_utils.py` file. You can follow the format to specify the input texts and target style labels. 

## Evaluation

### Download Pretrained Test Classifiers
You can download the evaluation style classifiers with the following command
```
python scripts/download_tst_classifiers.py --model_name [yelp-test, shakespeare-test-all]
```
And the fine-tuned language models for perplexity as below
```
python scripts/download_ppl_lms.py --model_name [yelp, shakespeare]
```
After that, you can run the evaluation script with the command below
```
cd evaluation
python run_eval.py \
    dataset=[yelp, shakespeare] \
    dataset_seed=[0, 1, 2 (skip for yelp)] \
    task_lm=[distilgpt2, gpt2, gpt2-medium, gpt2-large, gpt2-xl] \
    prompt_0_to_1=[learned prompt for the 0_to_1 direction (optional)] \
    prompt_1_to_0=[learned prompt for the 1_to_0 direction (optional)]
```
The outputs will be saved at `evaluation/outputs/[evaluation-date]/[evaluation-time]`
