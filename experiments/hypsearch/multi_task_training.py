#!/usr/bin/env python
# coding: utf-8

# # Multi-task Training with Hugging Face Transformers
# 

# ## Library setup
# 
# First up, we will install the libraries. 
# 
# <font color='red'>**Note: After running the following cell, you will need to restart your runtime for the installation to work properly.**</font>

# In[1]:


import numpy as np
import torch
import torch.nn as nn
import transformers
import datasets

import accelerate
from accelerate import Accelerator
from transformers import get_scheduler
import math 

import dataclasses
from torch.utils.data.dataloader import DataLoader
from transformers.data.data_collator import DefaultDataCollator, InputDataClass
from torch.utils.data.sampler import RandomSampler
from typing import List, Union, Dict
from transformers import default_data_collator
from torch.utils.data import Sampler

from tqdm.auto import tqdm
import os
import sklearn
import json

# ## Fetching our data
# 
# 

# In[2]:

def train(
        dataset_dict,
        mapping_column_dict,
        model_name = "models/roberta-large-mental-health-v1",
        learning_rate = 2e-6,
        epochs = 8,
        train_batch_size = 8,
        gradient_accumulation_steps = 1,
        weight_decay = 0.0,
        eps = 1e-8,
        num_warmup_steps = 200,
        scheduler_type = "linear",
        max_grad_norm = 1.0,
        output_path = './multitask/multitask_large',
        early_stopping_patience = 2,
        early_stopping_threshold = 0.025,
        sample_type = "main-uniform",
        max_length = 512,
        eval_steps = None,
        dropout_task = None
    ):

    # We can show one example from each task.

    # In[3]:


    for task_name, dataset in dataset_dict.items():
        print(task_name)
        print(dataset_dict[task_name]["train"][0])
        print()


    # In[4]:


    # columns of input and label
    #Fromat
    # [text_column, label_column]

    num_classes_dict = {}
    label_list_dict = {}

    for dataset in dataset_dict:
        label_list = dataset_dict[dataset]['train'].unique(mapping_column_dict[dataset][1])
        label_list.sort()
        label_list_dict[dataset] = label_list
        num_labels = len(label_list)
        num_classes_dict[dataset] = num_labels

        #Converts str label to int label encoding
        for split in dataset_dict[dataset]:
            dataset_dict[dataset][split] = dataset_dict[dataset][split].map(lambda x: {'label_encoding': label_list.index(x[mapping_column_dict[dataset][1]])})
        mapping_column_dict[dataset][1] = 'label_encoding'

    num_classes_dict


    # In[5]:


    label_list_dict


    # In[6]:


    mapping_column_dict


    # ## Creating a Multi-task Model
    # 
    # Next up, we are going to create a multi-task model. 
    # 
    # First, we define our `MultitaskModel` class:

    # In[7]:


    class MultitaskModel(transformers.PreTrainedModel):
        def __init__(self, encoder, taskmodels_dict):
            """
            Setting MultitaskModel up as a PretrainedModel allows us
            to take better advantage of Trainer features
            """
            super().__init__(transformers.PretrainedConfig())

            self.encoder = encoder
            self.taskmodels_dict = nn.ModuleDict(taskmodels_dict)

        @classmethod
        def create(cls, model_name, model_type_dict, model_config_dict):
            """
            This creates a MultitaskModel using the model class and config objects
            from single-task models. 

            We do this by creating each single-task model, and having them share
            the same encoder transformer.
            """
            shared_encoder = None
            taskmodels_dict = {}
            for task_name, model_type in model_type_dict.items():
                model = model_type.from_pretrained(
                    model_name, 
                    config=model_config_dict[task_name],
                )
                if shared_encoder is None:
                    shared_encoder = getattr(model, cls.get_encoder_attr_name(model))
                else:
                    setattr(model, cls.get_encoder_attr_name(model), shared_encoder)
                taskmodels_dict[task_name] = model
            return cls(encoder=shared_encoder, taskmodels_dict=taskmodels_dict)

        @classmethod
        def get_encoder_attr_name(cls, model):
            """
            The encoder transformer is named differently in each model "architecture".
            This method lets us get the name of the encoder attribute
            """
            model_class_name = model.__class__.__name__
            if model_class_name.startswith("Bert"):
                return "bert"
            elif model_class_name.startswith("Roberta"):
                return "roberta"
            elif model_class_name.startswith("Albert"):
                return "albert"
            elif model_class_name.startswith("Deberta"):
                return "deberta"
            else:
                raise KeyError(f"Add support for new model {model_class_name}")

        def forward(self, task_name, **kwargs):
            return self.taskmodels_dict[task_name](**kwargs)


    # As described above, the `MultitaskModel` class consists of only two components - the shared "encoder", a dictionary to the individual task models. Now, we can simply create the corresponding task models by supplying the invidual model classes and model configs. We will use Transformers' AutoModels to further automate the choice of model class given a model architecture (in our case, let's use `roberta-base`).

    # In[8]:

    model_type_dict = {}
    for dataset in dataset_dict:
        model_type_dict[dataset] = transformers.AutoModelForSequenceClassification

    model_config_dict = {}
    for dataset in dataset_dict:
        config = transformers.AutoConfig.from_pretrained(model_name, num_labels=num_classes_dict[dataset])
        if dropout_task is not None:
            if 'roberta' in model_name:
                config.classifier_dropout = dropout_task
            if 'deberta' in model_name:
                config.cls_dropout = dropout_task
        
        model_config_dict[dataset] = config

    multitask_model = MultitaskModel.create(
        model_name=model_name,
        model_type_dict=model_type_dict,
        model_config_dict=model_config_dict
    )


    # To confirm that all three task-models use the same encoder, we can check the data pointers of the respective encoders. In this case, we'll check that the word embeddings in each model all point to the same memory location.

    # In[9]:


    if 'roberta' in model_name:
        print('multitask', multitask_model.encoder.embeddings.word_embeddings.weight.data_ptr())
        for dataset in dataset_dict:
            print(dataset, multitask_model.taskmodels_dict[dataset].roberta.embeddings.word_embeddings.weight.data_ptr())
        #print(multitask_model.taskmodels_dict["external"].roberta.embeddings.word_embeddings.weight.data_ptr())
        # print(multitask_model.taskmodels_dict["commonsense_qa"].roberta.embeddings.word_embeddings.weight.data_ptr())
    else:
        print("Exercise for the reader: add a check for other model architectures =)")


    # ## Processing our task data
    # 

    # In[10]:


    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)


    # In[11]:

    convert_func_dict = {}

    def generate_convert_func(input_column, label_column):
        def convert_to_main_features(example_batch):
            inputs = example_batch[input_column]
            features = tokenizer.batch_encode_plus(
                inputs, max_length=max_length, pad_to_max_length=True
            )
            features["labels"] = example_batch[label_column]
            return features
        return convert_to_main_features

    for dataset in dataset_dict:
        convert_func_dict[dataset] = generate_convert_func(mapping_column_dict[dataset][0], mapping_column_dict[dataset][1])


    # Now that we have defined the above functions, we can use `dataset.map` to apply the functions over our entire datasets.

    # In[12]:

    columns_dict = {}
    for dataset in dataset_dict:
        columns_dict[dataset] = ['input_ids', 'attention_mask', 'labels']

    features_dict = {}
    for task_name, dataset in dataset_dict.items():
        features_dict[task_name] = {}
        for phase, phase_dataset in dataset.items():
            features_dict[task_name][phase] = phase_dataset.map(
                convert_func_dict[task_name],
                batched=True,
                load_from_cache_file=False,
            )
            print(task_name, phase, len(phase_dataset), len(features_dict[task_name][phase]))
            features_dict[task_name][phase].set_format(
                type="torch", 
                columns=columns_dict[task_name],
            )
            print(task_name, phase, len(phase_dataset), len(features_dict[task_name][phase]))


    # ## Preparing a multi-task data loader and Trainer

    # In[13]:

    class NLPDataCollator(DefaultDataCollator):
        """
        Extending the existing DataCollator to work with NLP dataset batches
        """
        def collate_batch(self, features: List[Union[InputDataClass, Dict]]) -> Dict[str, torch.Tensor]:
            first = features[0]
            if isinstance(first, dict):
              # NLP data sets current works presents features as lists of dictionary
              # (one per example), so we  will adapt the collate_batch logic for that
              if "labels" in first and first["labels"] is not None:
                  if first["labels"].dtype == torch.int64:
                      labels = torch.tensor([f["labels"] for f in features], dtype=torch.long)
                  else:
                      labels = torch.tensor([f["labels"] for f in features], dtype=torch.float)
                  batch = {"labels": labels}
              for k, v in first.items():
                  if k != "labels" and v is not None and not isinstance(v, str):
                      batch[k] = torch.stack([f[k] for f in features])
              return batch
            else:
              # otherwise, revert to using the default collate_batch
              return default_data_collator().collate_batch(features)


    class StrIgnoreDevice(str):
        """
        This is a hack. The Trainer is going call .to(device) on every input
        value, but we need to pass in an additional `task_name` string.
        This prevents it from throwing an error
        """
        def to(self, device):
            return self


    class DataLoaderWithTaskname:
        """
        Wrapper around a DataLoader to also yield a task name
        """
        def __init__(self, task_name, data_loader):
            self.task_name = task_name
            self.data_loader = data_loader

            self.batch_size = data_loader.batch_size
            self.dataset = data_loader.dataset

        def __len__(self):
            return len(self.data_loader)

        def __iter__(self):
            for batch in self.data_loader:
                batch["task_name"] = StrIgnoreDevice(self.task_name)
                yield batch


    class MultitaskDataloader:
        """
        Data loader that combines and samples from multiple single-task
        data loaders.
        """
        def __init__(self, dataloader_dict, sample_type='size-proportional'):
            self.dataloader_dict = dataloader_dict
            self.num_batches_dict = {
                task_name: len(dataloader) 
                for task_name, dataloader in self.dataloader_dict.items()
            }
            self.task_name_list = list(self.dataloader_dict)
            self.dataset = [None] * sum(
                len(dataloader.dataset) 
                for dataloader in self.dataloader_dict.values()
            )

            self.sample_type = sample_type

        def _get_infinite_generator(self, dataloader):
            while True:
                for data in dataloader:
                    yield data

        def __len__(self):
            if self.sample_type == "size-proportional":
                return sum(self.num_batches_dict.values())
            elif self.sample_type == "main-uniform":
                return int(self.num_batches_dict['main']*len(self.num_batches_dict))
            else:
                raise Exception(f"Invalid sample_type {self.sample_type} for class MultitaskDataloader")

        def __iter__(self):
            """
            For each batch, sample a task, and yield a batch from the respective
            task Dataloader.

            We use size-proportional sampling, but you could easily modify this
            to sample from some-other distribution.
            """
            if self.sample_type == "size-proportional":
                dataloader_iter_dict = {
                    task_name: iter(dataloader) 
                    for task_name, dataloader in self.dataloader_dict.items()
                }

                task_choice_list = []

                for i, task_name in enumerate(self.task_name_list):
                    task_choice_list += [i] * self.num_batches_dict[task_name]

                task_choice_list = np.array(task_choice_list)
                np.random.shuffle(task_choice_list)

                for task_choice in task_choice_list:
                    task_name = self.task_name_list[task_choice]
                    yield next(dataloader_iter_dict[task_name]) 
            elif self.sample_type == "main-uniform":
                if not hasattr(self, 'dataloader_iter_dict'):
                    self.dataloader_iter_dict = {
                        task_name: self._get_infinite_generator(dataloader) 
                        for task_name, dataloader in self.dataloader_dict.items()
                        if task_name != 'main'
                    }
                self.dataloader_iter_dict['main'] = iter(self.dataloader_dict['main'])

                task_choice_list = []

                for i, task_name in enumerate(self.task_name_list):
                    task_choice_list += [i] * self.num_batches_dict['main']

                task_choice_list = np.array(task_choice_list)
                np.random.shuffle(task_choice_list)

                for task_choice in task_choice_list:
                    task_name = self.task_name_list[task_choice]
                    yield next(self.dataloader_iter_dict[task_name])
            else:
                raise Exception(f"Invalid sample_type {self.sample_type} for class MultitaskDataloader")


    def get_single_train_dataloader(task_name, train_dataset, train_batch_size=32, data_collator=NLPDataCollator()):
            """
            Create a single-task data loader that also yields task names
            """
            if train_dataset is None:
                raise ValueError("Trainer: training requires a train_dataset.")

            train_sampler = (
                RandomSampler(train_dataset)
            )

            data_loader = DataLoaderWithTaskname(
                task_name=task_name,
                data_loader=DataLoader(
                  train_dataset,
                  batch_size=train_batch_size,
                  sampler=train_sampler,
                  collate_fn=data_collator.collate_batch,
                ),
            )

            return data_loader


    # In[15]:


    optimizer_grouped_parameters = [
            {
                "params": [p for n, p in multitask_model.named_parameters()],
                "weight_decay": weight_decay,
            },

        ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=eps)

    train_dataset = {
        task_name: dataset["train"] 
        for task_name, dataset in features_dict.items()
    }
    train_dataloader = MultitaskDataloader({
                task_name: get_single_train_dataloader(task_name, task_dataset, train_batch_size=train_batch_size)
                for task_name, task_dataset in train_dataset.items()
            }, sample_type=sample_type)


    # In[16]:


    eval_dataset = {
        "main": features_dict["main"]["validation"] 
    }
    eval_dataloader = MultitaskDataloader({
                "main": get_single_train_dataloader("main", features_dict["main"]["validation"], train_batch_size=train_batch_size )
            })


    # In[17]:


    num_training_steps = epochs * math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    lr_scheduler = get_scheduler(
        name=scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )


    # ## Time to train!
    # 

    # In[18]:

    #device = torch.device("cuda:2") if torch.cuda.is_available() else torch.device("cpu")

    accelerator = Accelerator(device_placement=True, gradient_accumulation_steps=gradient_accumulation_steps, log_with="wandb", mixed_precision="fp16")
    tags = os.getenv('WANDB_PROJECT', "depsign_multitask")
    accelerator.init_trackers(
        project_name=os.getenv('WANDB_PROJECT', "depsign_multitask"), 
        config={
            "model_name": model_name,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "train_batch_size": train_batch_size,
            "gradient_accumulation_steps":  gradient_accumulation_steps,
            "weight_decay":  weight_decay,
            "eps": eps,
            "num_warmup_steps": num_warmup_steps,
            "scheduler_type": scheduler_type,
            "max_grad_norm": max_grad_norm,
            "output_path": output_path,
            "max_seq_length": max_length,
            "sample_type": sample_type
        },
        init_kwargs={
            "wandb": {
                "entity": os.getenv('WANDB_ENTITY', "dlb-depsign"),
                "tags": os.getenv('WANDB_TAGS', "cds").split(',')
            }
        }
    )

    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
            multitask_model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
        )

    progress_bar = tqdm(range(num_training_steps), disable=not accelerator.is_local_main_process)

    #model = accelerator.prepare(multitask_model.train())
    #multitask_model.train().to(accelerator.device)


    # In[ ]:


    last_main_metric = 0

    avg_loss = 0
    global_step = 0

    main_task_loss = 0
    main_task_step = 0

    other_tasks_loss = 0
    other_tasks_step = 0

    patience_count = 0
    
    #_train_dataloader = train_dataloader
    #if eval_steps is not None:
    #    def _get_ifinite_dataloader():
    #        while True:
    #            for batch in train_dataloader:
    #                yield batch
    #    _train_dataloader = _get_ifinite_dataloader()
    
    epoch = 0
    _train_dataloader = iter(train_dataloader)
    
    while epoch < epochs:
        model.train()
        eval_break = False
        for step, batch in enumerate(_train_dataloader):
            with accelerator.accumulate(model):
                task_name = batch["task_name"]
                input_ids = batch["input_ids"].to(accelerator.device)
                attention_mask = batch["attention_mask"].to(accelerator.device)
                labels = batch["labels"].to(accelerator.device)
                outputs = model.taskmodels_dict[task_name](input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                #accelerator.print(loss)
                #accelerator.log({"train_loss": 1.12}, step=step)
                avg_loss += loss.item()
                if task_name == 'main':
                    main_task_loss += loss.item()
                    main_task_step += 1
                else:
                    other_tasks_loss += loss.item()
                    other_tasks_step += 1
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                #if step % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

            if accelerator.sync_gradients and accelerator.is_main_process:
                global_step += 1
                current_loss = accelerator.gather(avg_loss / global_step)
                accelerator.log({
                    "train_loss": current_loss,
                    "main_task_step": accelerator.gather(main_task_step),
                    "other_tasks_step": accelerator.gather(other_tasks_step),
                    "main_task_train_loss": accelerator.gather(main_task_loss / (main_task_step if main_task_step != 0 else 1)),
                    "other_tasks_train_loss": accelerator.gather(other_tasks_loss / (other_tasks_step if other_tasks_step != 0 else 1)),
                    'current_learning_rate': accelerator.gather(lr_scheduler.get_last_lr())
                }, step=global_step)
            
                if eval_steps is not None and global_step % eval_steps == 0:
                    eval_break = True
                    break
                #accelerator.print("train_loss:", current_loss)
        
        model.eval() 
        all_predictions = {}
        all_references = {}
        eval_loss = 0
        accelerator.print(f'Evaluating... Epoch: {epoch}, Global Step: {global_step}')
        for step, batch in enumerate(eval_dataloader):
            task_name = batch["task_name"]
            if task_name != "main":
                continue
            # print(task_name)
            input_ids = batch["input_ids"].to(accelerator.device)
            attention_mask = batch["attention_mask"].to(accelerator.device)
            labels = batch["labels"].to(accelerator.device)
            # print(len(labels), len(input_ids))
            #outputs = multitask_model.taskmodels_dict[task_name](input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            with torch.no_grad():
                outputs = model.taskmodels_dict[task_name](input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                # print(outputs.logits)
            loss = outputs.loss
            eval_loss += loss.item()
            predictions = outputs.logits.argmax(dim=-1) 
            # predictions, references = accelerator.gather((predictions, batch["labels"]))
            # print(predictions)
            predictions = predictions.cpu().tolist()
            references = labels.cpu().tolist()

            if task_name in all_predictions:
                all_predictions[task_name].extend(predictions)
            else:
                all_predictions[task_name] = predictions

            if task_name in all_references:
                all_references[task_name].extend(references)
            else:
                all_references[task_name] = references

        metrics = sklearn.metrics.classification_report(all_references["main"], all_predictions["main"], digits=4)
        accelerator.print(metrics)

        main_metric = sklearn.metrics.f1_score(all_references["main"], all_predictions["main"], average='macro')
        accelerator.log({"eval_loss": accelerator.gather(eval_loss / len(eval_dataloader)), "eval_f1 (macro)": main_metric}, step=global_step)

        if main_metric - last_main_metric < early_stopping_threshold:
            patience_count += 1

        if main_metric > last_main_metric:
            accelerator.print(f'Saving new best model at epoch {epoch}, global_step {global_step}')
            accelerator.wait_for_everyone()
            unwrap_model = accelerator.unwrap_model(model)

            unwrap_model.save_pretrained(output_path)
            accelerator.save(unwrap_model.state_dict(), os.path.join(output_path, "state_dict.bin"))
            last_main_metric = main_metric
            with open(os.path.join(output_path, "report.txt"), 'w') as f:
                f.write(metrics+'\n')
                f.write(f'Epoch: {epoch}, Global Step: {global_step}')
            
            with open(os.path.join(output_path, "all_results.json"), 'w') as f:
                json.dump({"eval_f1 (macro)": main_metric}, f, indent=4)

        if patience_count >= early_stopping_patience:
            accelerator.print(f'Early stopping paticience {patience_count} reached, stop training...')
            break
        
        if not eval_break:
            _train_dataloader = iter(train_dataloader)
            epoch += 1

    accelerator.end_training()

def main(args):
    dataset_dict = {
        "main": datasets.load_dataset('csv', data_files={"train": args.train_file, "validation": args.validation_file}),
        "cds": datasets.load_dataset('csv', data_files={
                "train": "https://huggingface.co/datasets/hungchiayu/cds-dataset2-depression/resolve/main/trainDepression.csv",
                "validation": "https://huggingface.co/datasets/hungchiayu/cds-dataset2-depression/resolve/main/testDepression.csv"
            }
        ),
    }
    
    mapping_column_dict = {
        "main": ['text data', 'label'],
        "cds":  ['text', 'label_name']
    }
    
    train(
        dataset_dict,
        mapping_column_dict,
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        eps=args.eps,
        num_warmup_steps=args.num_warmup_steps,
        scheduler_type=args.scheduler_type,
        max_grad_norm=args.max_grad_norm,
        output_path=args.output_dir,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=args.early_stopping_threshold,
        sample_type=args.sample_type,
        max_length=args.max_length,
        eval_steps=args.eval_steps,
        dropout_task=args.dropout_task,
    )

    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
                    prog='Multi Task Training',
                    description='Train a multi task model with transformers')
    
    parser.add_argument(
        "-m",
        "--model_name",
        required=True,
        type=str
    )
    
    parser.add_argument(
        "-o",
        "--output_dir",
        required=True,
        type=str
    )
    
    parser.add_argument(
        "--train_file",
        type=str,
        required=True
    )
    
    parser.add_argument(
        "--validation_file",
        type=str,
        required=True
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-6
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=5
    )
    
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=8
    )
    
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1
    )
    
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0
    )
    
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-8
    )
    
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=200
    )
    
    parser.add_argument(
        "--scheduler_type",
        type=str,
        default="linear"
    )
    
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0
    )
    
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=2
    )
    
    parser.add_argument(
        "--early_stopping_threshold",
        type=float,
        default=0.025
    )
    
    parser.add_argument(
        "--sample_type",
        type=str,
        default="main-uniform"
    )
    
    parser.add_argument(
        "--max_length",
        type=int,
        default=512
    )
    
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=None
    )
    
    parser.add_argument(
        "--dropout_task",
        type=float,
        default=None
    )
    
    #Not used, backwards compability huggingface script
    parser.add_argument(
        "--overwrite_output_dir",
        action='store_true',
        default=False
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )
    
    args = parser.parse_args()
    print(args)
    main(args)
