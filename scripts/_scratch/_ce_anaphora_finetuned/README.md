---
tags:
- sentence-transformers
- cross-encoder
- reranker
- generated_from_trainer
- dataset_size:568
- loss:BinaryCrossEntropyLoss
base_model: cross-encoder/ms-marco-MiniLM-L6-v2
pipeline_tag: text-ranking
library_name: sentence-transformers
---

# CrossEncoder based on cross-encoder/ms-marco-MiniLM-L6-v2

This is a [Cross Encoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html) model finetuned from [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) using the [sentence-transformers](https://www.SBERT.net) library. It computes scores for pairs of texts, which can be used for text reranking and semantic search.

## Model Details

### Model Description
- **Model Type:** Cross Encoder
- **Base model:** [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) <!-- at revision c5ee24cb16019beea0893ab7796b1df96625c6b8 -->
- **Maximum Sequence Length:** 512 tokens
- **Number of Output Labels:** 1 label
- **Supported Modality:** Text
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Documentation:** [Cross Encoder Documentation](https://www.sbert.net/docs/cross_encoder/usage/usage.html)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Cross Encoders on Hugging Face](https://huggingface.co/models?library=sentence-transformers&other=cross-encoder)

### Full Model Architecture

```
CrossEncoder(
  (0): Transformer({'transformer_task': 'sequence-classification', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'logits'}}, 'module_output_name': 'scores', 'architecture': 'BertForSequenceClassification'})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import CrossEncoder

# Download from the 🤗 Hub
model = CrossEncoder("cross_encoder_model_id")
# Get scores for pairs of inputs
pairs = [
    ['is this the same as capstan drives?', "I think I'm not going to use the hybrid leg paper but try to mimic the joint configuration of the lima robot in the paper with the motors from dfrobot. It seems I may need the robot fully CAD designed. Would Fusion360 be sufficient or does ReActor use a different model output?"],
    ['I need to state that the robot bear is not quadrupedal. It walks on two legs. It crawls to go up steps. The Lima robot they are using seems in the paper seems like the perfect size and range of motion for this project', 'Actually I think at this cost we can afford to build two phone cameras into the eyes and have the vl53L9cx sit ajna position between them. We can then postprocess any video into stereo in remote hardware and get better eye tracking when the user interacts with the robot'],
    ['Essentially. The optimalism paper talks about legacy and death. We probably should have a card in there that causes a player to receive and inheritance or the opportunity to buy out an estate as a result of being a benefactor', 'A lot of what you picked up from vantage is good. What I really was interested is if you understood its game play loop. The players are placed in a world by drawing map tiles (cards), The map tiles have options of what the player can do, features in the landscape, and directions they can go to reach other map tiles with different options. When they take actions they draw cards and read from a guide book that defines what happens and a choose your own adventure styles set of options. We should do something similar to this. but because we are modeling the real world we can use real map tiles and extract features to make up what is possible within a geographic location such as this. Since we need mgrs data to track commerce they will become map tiles. Players should build their society through history with ages and each age brings greater technology more jobs and more actions that they can take. It will also bring modes of transportation that can increase the distance a player can travel in a day. Making time progressively more efficient to be used by players to accomplish more things encouraging cooperation to optimize use of time'],
    ["I guess I do now need to figure out how a ledger works for big-time users processing large scale many-to-one transaction against their account. I really like how skycoin does their consensus but I don't know if that is feasible in practice", 'Well let me add this caveat. If I\'m walmart my ledger may need to be spread across multiple processors. So maybe they share on "CRAB" that they collaborate with each other or maybe the ledgers is sharded some how between them. I just know the solution needs to work in the case of small-time Alice and big-time Walmart and maintain compatibility'],
    ['D sounds like the most relevant. This plays into what is a players individual statistics that defines them and their possibilities', "Players also live and die in finite periods of time and take on the role of new people. The ages don't move in one life time. They could also die at any time from environment issues, the results of their actions, other players, scarcity of resources including free time and stress. Players also are born with talents and abilities that make them better or worse at things. This will incentivize players to specialize and cooperate to achieve moving society forward "],
]
scores = model.predict(pairs)
print(scores)
# [-6.1484 -2.0801 -2.6387  1.9766 -1.1201]

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    'is this the same as capstan drives?',
    [
        "I think I'm not going to use the hybrid leg paper but try to mimic the joint configuration of the lima robot in the paper with the motors from dfrobot. It seems I may need the robot fully CAD designed. Would Fusion360 be sufficient or does ReActor use a different model output?",
        'Actually I think at this cost we can afford to build two phone cameras into the eyes and have the vl53L9cx sit ajna position between them. We can then postprocess any video into stereo in remote hardware and get better eye tracking when the user interacts with the robot',
        'A lot of what you picked up from vantage is good. What I really was interested is if you understood its game play loop. The players are placed in a world by drawing map tiles (cards), The map tiles have options of what the player can do, features in the landscape, and directions they can go to reach other map tiles with different options. When they take actions they draw cards and read from a guide book that defines what happens and a choose your own adventure styles set of options. We should do something similar to this. but because we are modeling the real world we can use real map tiles and extract features to make up what is possible within a geographic location such as this. Since we need mgrs data to track commerce they will become map tiles. Players should build their society through history with ages and each age brings greater technology more jobs and more actions that they can take. It will also bring modes of transportation that can increase the distance a player can travel in a day. Making time progressively more efficient to be used by players to accomplish more things encouraging cooperation to optimize use of time',
        'Well let me add this caveat. If I\'m walmart my ledger may need to be spread across multiple processors. So maybe they share on "CRAB" that they collaborate with each other or maybe the ledgers is sharded some how between them. I just know the solution needs to work in the case of small-time Alice and big-time Walmart and maintain compatibility',
        "Players also live and die in finite periods of time and take on the role of new people. The ages don't move in one life time. They could also die at any time from environment issues, the results of their actions, other players, scarcity of resources including free time and stress. Players also are born with talents and abilities that make them better or worse at things. This will incentivize players to specialize and cooperate to achieve moving society forward ",
    ]
)
# [{'corpus_id': ..., 'score': ...}, {'corpus_id': ..., 'score': ...}, ...]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 568 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 100 samples:
  |          | sentence_0                                                                         | sentence_1                                                                         | label                                                          |
  |:---------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type     | string                                                                             | string                                                                             | float                                                          |
  | modality | text                                                                               | text                                                                               |                                                                |
  | details  | <ul><li>min: 6 tokens</li><li>mean: 58.86 tokens</li><li>max: 512 tokens</li></ul> | <ul><li>min: 5 tokens</li><li>mean: 73.65 tokens</li><li>max: 395 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.18</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                     | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | label            |
  |:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>is this the same as capstan drives?</code>                                                                                                                                                                                               | <code>I think I'm not going to use the hybrid leg paper but try to mimic the joint configuration of the lima robot in the paper with the motors from dfrobot. It seems I may need the robot fully CAD designed. Would Fusion360 be sufficient or does ReActor use a different model output?</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | <code>0.0</code> |
  | <code>I need to state that the robot bear is not quadrupedal. It walks on two legs. It crawls to go up steps. The Lima robot they are using seems in the paper seems like the perfect size and range of motion for this project</code>         | <code>Actually I think at this cost we can afford to build two phone cameras into the eyes and have the vl53L9cx sit ajna position between them. We can then postprocess any video into stereo in remote hardware and get better eye tracking when the user interacts with the robot</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | <code>0.0</code> |
  | <code>Essentially. The optimalism paper talks about legacy and death. We probably should have a card in there that causes a player to receive and inheritance or the opportunity to buy out an estate as a result of being a benefactor</code> | <code>A lot of what you picked up from vantage is good. What I really was interested is if you understood its game play loop. The players are placed in a world by drawing map tiles (cards), The map tiles have options of what the player can do, features in the landscape, and directions they can go to reach other map tiles with different options. When they take actions they draw cards and read from a guide book that defines what happens and a choose your own adventure styles set of options. We should do something similar to this. but because we are modeling the real world we can use real map tiles and extract features to make up what is possible within a geographic location such as this. Since we need mgrs data to track commerce they will become map tiles. Players should build their society through history with ages and each age brings greater technology more jobs and more actions that they can take. It will also bring modes of transportation that can increase the distance a player can travel ...</code> | <code>0.0</code> |
* Loss: [<code>BinaryCrossEntropyLoss</code>](https://sbert.net/docs/package_reference/cross_encoder/losses.html#binarycrossentropyloss) with these parameters:
  ```json
  {
      "activation_fn": "torch.nn.modules.linear.Identity",
      "pos_weight": null
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `fp16`: True
- `per_device_eval_batch_size`: 16

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 16
- `num_train_epochs`: 3
- `max_steps`: -1
- `learning_rate`: 5e-05
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_steps`: 0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `optim_target_modules`: None
- `gradient_accumulation_steps`: 1
- `average_tokens_across_devices`: True
- `max_grad_norm`: 1
- `label_smoothing_factor`: 0.0
- `bf16`: False
- `fp16`: True
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `use_cache`: False
- `neftune_noise_alpha`: None
- `torch_empty_cache_steps`: None
- `auto_find_batch_size`: False
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `include_num_input_tokens_seen`: no
- `log_level`: passive
- `log_level_replica`: warning
- `disable_tqdm`: False
- `project`: huggingface
- `trackio_space_id`: None
- `trackio_bucket_id`: None
- `trackio_static_space_id`: None
- `per_device_eval_batch_size`: 16
- `prediction_loss_only`: True
- `eval_on_start`: False
- `eval_do_concat_batches`: True
- `eval_use_gather_object`: False
- `eval_accumulation_steps`: None
- `include_for_metrics`: []
- `batch_eval_metrics`: False
- `save_only_model`: False
- `save_on_each_node`: False
- `enable_jit_checkpoint`: False
- `push_to_hub`: False
- `hub_private_repo`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_always_push`: False
- `hub_revision`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `restore_callback_states_from_checkpoint`: False
- `full_determinism`: False
- `seed`: 42
- `data_seed`: None
- `use_cpu`: False
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `dataloader_prefetch_factor`: None
- `remove_unused_columns`: True
- `label_names`: None
- `train_sampling_strategy`: random
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `ddp_static_graph`: None
- `ddp_backend`: None
- `ddp_timeout`: 1800
- `fsdp`: []
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `deepspeed`: None
- `debug`: []
- `skip_memory_metrics`: True
- `do_predict`: False
- `resume_from_checkpoint`: None
- `warmup_ratio`: None
- `local_rank`: -1
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Time
- **Training**: 3.3 seconds

### Framework Versions
- Python: 3.14.5
- Sentence Transformers: 5.6.0
- Transformers: 5.6.2
- PyTorch: 2.12.1+cu130
- Accelerate: 1.14.0
- Datasets: 5.0.0
- Tokenizers: 0.22.2

## Additional Resources

- [Training and Finetuning Reranker Models with Sentence Transformers](https://huggingface.co/blog/train-reranker): the end-to-end guide for training or finetuning Cross Encoder (reranker) models.
- [Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/multimodal-sentence-transformers): use text, image, audio, and video reranker models through the same API.
- [Training and Finetuning Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/train-multimodal-sentence-transformers): training multimodal Cross Encoders.

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->