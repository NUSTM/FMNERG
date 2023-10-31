# -*- coding: utf-8 -*-

import argparse
import os
import logging
import time
import pickle
from tqdm import tqdm
import datetime

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from transformers import AdamW, T5ForConditionalGeneration, T5Tokenizer, T5Config
# from transformers import BertTokenizer, EncoderDecoderModel
from transformers import get_linear_schedule_with_warmup

from VisionT5 import VisionT5
from data_utils import VisualDataset
from data_utils import read_line_examples_from_file
from eval_utils import compute_scores, turn_vis_similarities_to_vis_pred

# from more_eval import eval_ouput_file_FMNERG, elaluate_all, elaluate_EEG

logger = logging.getLogger(__name__)


def init_args():
    parser = argparse.ArgumentParser()
    # basic settings
    parser.add_argument("--task", default='GMNER', type=str, required=True)
    parser.add_argument("--dataset", default='twitter17', type=str, required=True,  # rest15
                        help="The name of the dataset, selected from: [rest15, rest16]")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--imgtag_path_rcnn", default=None, type=str,
                        help="the path of img tag path")
    parser.add_argument("--imgtag_path_vinvl", default=None, type=str,
                        help="the path of img tag path")
    parser.add_argument("--imgtag_path_anp", default=None, type=str,
                        help="the path of img tag path")
    parser.add_argument("--model_name_or_path", default='./t5-base', type=str,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev/test set.")
    parser.add_argument("--do_direct_eval", action='store_true',
                        help="Whether to run eval on the dev/test set.")
    parser.add_argument("--do_inference", action='store_true',
                        help="Whether to run inference with trained checkpoints")
    parser.add_argument("--has_caption", action='store_true',
                        help="Whether the input contain caption")
    parser.add_argument("--has_tag", action='store_true',
                        help="Whether the input contain caption")
    parser.add_argument("--output_pred", action='store_true',
                        help="Whether to write the preds")

    # other parameters
    parser.add_argument("--vinvl_region_number", default=36, type=int)
    parser.add_argument("--anp_number", default=0, type=int)
    parser.add_argument("--loss_distribution", default=1, type=int)
    parser.add_argument("--max_seq_length", default=200, type=int)
    parser.add_argument("--n_gpu", default=0)
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=3e-4, type=float)
    parser.add_argument("--num_train_epochs", default=30, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--num_beams", default=1, type=int)
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--checkpoint', default='./outputs/', type=str)

    # training details
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--warmup_steps", default=0.0, type=float)

    # add_vinVL
    parser.add_argument("--img_path_vinvl", default='/root/data2/twitter_images/twitterGMNER_vinvl_extract36', type=str)
    parser.add_argument("--image_annotation_path", default='/root/jmwang/brick/NER-CLS-VG/data/version2/xml', type=str)
    parser.add_argument("--use_visual_feats", default=True, type=bool)
    parser.add_argument("--coarse_grained_auxiliary", default=True, type=bool)

    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--feat_dim', type=float, default=2048)
    parser.add_argument('--pos_dim', type=float, default=36)
    parser.add_argument('--debug', type=bool, default=False)

    args = parser.parse_args()

    # set up output dir which looks like './outputs/rest15/'
    if not os.path.exists(f'./outputs/'):
        os.mkdir(f'./outputs/')

    now = int(round(time.time()))
    local_time = time.strftime('%Y%m%d-%H:%M:%S', time.localtime(now))
    if args.do_inference is False:
        notes = str('debug_' if args.debug else '') + '_SSEP_' + str(args.num_train_epochs) + 'epoch_' + str(
            args.vinvl_region_number) + 'box_' + str(args.seed) + '_' + str(args.learning_rate)
    else:
        notes = 'only_inference'  # + # args.checkpoint

    output_dir = f"./outputs/{notes}_{local_time}"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    args.output_dir = output_dir

    return args


def get_dataset(tokenizer, data_type, args):
    return VisualDataset(
        tokenizer=tokenizer,
        data_dir=args.data_dir,
        data_set=args.dataset,
        data_type=data_type,
        max_len=args.max_seq_length,
        vinvl_region_number=args.vinvl_region_number,
        img_path_vinvl=args.img_path_vinvl,
        image_annotation_path=args.image_annotation_path,
        use_visual_feats=args.use_visual_feats,
        coarse_grained_auxiliary=args.coarse_grained_auxiliary
    )


class T5FineTuner(pl.LightningModule):
    """
    Fine tune a pre-trained T5 model
    """

    def __init__(self, hparams, tfm_model, tokenizer):
        super().__init__()
        self.hparams = hparams
        self.model = tfm_model
        self.tokenizer = tokenizer

    def is_logger(self):
        return True

    def forward(self, input_ids, attention_mask=None, vis_feats=None, vis_attention_mask=None, img_label=None,
                decoder_input_ids=None, decoder_attention_mask=None, labels=None):
        return self.model(
            input_ids,
            attention_mask=attention_mask,

            vis_feats=vis_feats,
            vis_attention_mask=vis_attention_mask,
            img_label=img_label,

            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

    def _step(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        vis_feats = batch["vis_feats"]
        vis_attention_mask = batch["vis_attention_mask"]
        img_label = batch["img_label"]

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],

            vis_feats=vis_feats,
            vis_attention_mask=vis_attention_mask,
            img_label=img_label,

            labels=lm_labels,
            decoder_attention_mask=batch['target_mask']
        )

        loss = outputs[0]
        return loss

    def training_step(self, batch, batch_idx):
        start = datetime.datetime.now()  # #
        loss = self._step(batch)
        end = datetime.datetime.now()  # #
        diff_time = end - start  # #
        step_time = torch.tensor(diff_time.microseconds, device=loss.device)
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs, "step_time": step_time}  # #

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        # import pdb;pdb.set_trace()
        sum_step_time = torch.stack([x['step_time'] for x in outputs]).sum() * 1e-6  # # jmwang
        tensorboard_logs = {"avg_train_loss": avg_train_loss, "sum_step_time": sum_step_time}  ##jmwang
        return {"avg_train_loss": avg_train_loss, "sum_step_time": sum_step_time, "log": tensorboard_logs,
                'progress_bar': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def configure_optimizers(self):
        """ Prepare optimizer and schedule (linear warmup and decay) """
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(self,
                       epoch,
                       batch_idx,
                       optimizer,
                       optimizer_idx=0,
                       optimizer_closure=None,
                       on_tpu=False,
                       using_lbfgs=False):
        optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.4f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}
        return tqdm_dict

    def train_dataloader(self):
        train_dataset = get_dataset(tokenizer=self.tokenizer, data_type="train", args=self.hparams)
        dataloader = DataLoader(train_dataset, batch_size=self.hparams.train_batch_size,
                                drop_last=True, shuffle=True, num_workers=4)
        # dataloader.object_detection_faults = train_dataset.object_detection_faults
        t_total = (
                (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, len(self.hparams.n_gpu))))
                // self.hparams.gradient_accumulation_steps
                * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        val_dataset = get_dataset(tokenizer=self.tokenizer, data_type="dev", args=self.hparams)
        val_dataloader = DataLoader(val_dataset, batch_size=self.hparams.eval_batch_size, num_workers=4)
        # val_dataloader.object_detection_faults = val_dataset.object_detection_faults
        return val_dataloader


class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        logger.info("***** Validation results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
        # Log results
        for key in sorted(metrics):
            if key not in ["log", "progress_bar"]:
                logger.info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer, pl_module):
        logger.info("***** Test results *****")

        if pl_module.is_logger():
            metrics = trainer.callback_metrics

        # Log and save results to file
        output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
        with open(output_test_results_file, "w") as writer:
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    logger.info("{} = {}\n".format(key, str(metrics[key])))
                    writer.write("{} = {}\n".format(key, str(metrics[key])))


def create_config(args):
    config = T5Config.from_pretrained(args.model_name_or_path)

    config.feat_dim = args.feat_dim
    config.pos_dim = args.pos_dim

    config.dropout_rate = args.dropout
    config.dropout = args.dropout
    config.attention_dropout = args.dropout
    config.activation_dropout = args.dropout
    config.vinvl_region_number = args.vinvl_region_number

    return config


def evaluate(data_loader, model, num_beams, vinvl_region_number, object_detection_faults):
    """
    Compute scores given the predictions and gold labels
    """
    device = torch.device(f'cuda:{args.n_gpu}')
    model.model.to(device)

    model.model.eval()

    outputs, targets, vis_preds, img_labels, img_ids = [], [], [], [], []
    all_pred_sentence_number = 0
    all_wrong_match_sentence_number = 0

    for batch in tqdm(data_loader):
        # need to push the data to device
        outs, vis_similarities = model.model.generate_VisionT5(
            input_ids=batch['source_ids'].to(device),
            attention_mask=batch['source_mask'].to(device),
            vis_feats=batch["vis_feats"].to(device),
            vis_attention_mask=batch["vis_attention_mask"].to(device),
            max_length=200, num_beams=num_beams,
            vinvl_region_number=vinvl_region_number)  # num_beams=8, early_stopping=True)

        dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
        target = [tokenizer.decode(ids, skip_special_tokens=True) for ids in batch["target_ids"]]
        vis_pred = turn_vis_similarities_to_vis_pred(vis_similarities, outs, vinvl_region_number).tolist()
        img_label = batch['img_label']
        img_id = batch['img_id']
        # object_detection_fault = batch['object_detection_fault']

        mask_img_label = []
        for batch_i, label in enumerate(img_label):
            this_batch = []
            for entity in label:
                if sum(entity) == 0:
                    this_batch.append(False)
                # elif sum(entity) == -100:   # object detection fault
                #     this_batch.append(True)
                else:
                    this_batch.append(True)

            mask_img_label.append(this_batch)
        mask_img_label = torch.tensor(mask_img_label)
        img_label = img_label[mask_img_label]
        # img_label = torch.where(torch.tensor(img_label < 0).to(device), torch.tensor(0.0).to(device), img_label.to(device))    # object_detection_faults
        img_label = [label.nonzero().squeeze(dim=-1).tolist() for label in img_label]

        vis_preds.extend(vis_pred)
        img_labels.extend(img_label)
        outputs.extend(dec)
        targets.extend(target)
        img_ids.extend(img_id)
        # object_detection_faults.extend(object_detection_fault)

    """
    print("\nPrint some results to check the sanity of generation method:", '\n', '-'*30)
    for i in [1, 5, 25, 42, 50]:
        try:
            print(f'>>Target    : {targets[i]}')
            print(f'>>Generation: {outputs[i]}')
        except UnicodeEncodeError:
            print('Unable to print due to the coding error')
    print()
    """

    scores, all_labels, all_preds, all_coarse_labels, all_coarse_preds = compute_scores(outputs, targets, vis_preds,
                                                                                        img_labels, img_ids,
                                                                                        object_detection_faults,
                                                                                        all_pred_sentence_number,
                                                                                        all_wrong_match_sentence_number)
    # scores, all_labels, all_preds, all_coarse_labels, all_coarse_preds = compute_scores(outputs, targets, vis_preds, img_labels, img_ids, object_detection_faults)

    # path_dir = f'{args.output_dir}/results-{args.learning_rate}-{args.vinvl_region_number}.txt'
    if args.output_pred:
        with open(
                f"{args.output_dir}/results-{args.learning_rate}-{args.vinvl_region_number}-{args.seed}.txt",
                'w') as f:
            for i in range(len(all_labels)):
                f.write("fine     GT: " + str(all_labels[i]) + "\n")
                f.write("fine   Pred: " + str(all_preds[i]) + "\n")
                f.write("coarse   GT: " + str(all_coarse_labels[i]) + "\n")
                f.write("coarse Pred: " + str(all_coarse_preds[i]) + "\n")
                f.write("\n")

    # pickle.dump(results, open(f"{args.output_dir}/results-{args.dataset}.pickle", 'wb'))

    return scores


# initialization
args = init_args()
print("\n", "=" * 30, f"NEW EXP: ASQP on {args.dataset}", "=" * 30, "\n")
seed_everything(args.seed)  ## jmwang add
# sanity check
# show one sample to check the code and the expected output
kwargs = {}
tokenizer = T5Tokenizer.from_pretrained(
    args.model_name_or_path,
    max_length=200,
    **kwargs
)
print(f"Here is an example (from the dev set):")

dataset = get_dataset(tokenizer=tokenizer, data_type='dev', args=args)
data_sample = dataset[7]  # a random data sample
print('Input :', tokenizer.decode(data_sample['source_ids'], skip_special_tokens=True))
print('Output:', tokenizer.decode(data_sample['target_ids'], skip_special_tokens=True))

# training process
if args.do_train:
    print("\n****** Conduct Training ******")

    # initialize the T5 model
    print('start tfm_model')
    # tfm_model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
    config = create_config(args)
    tfm_model = VisionT5.from_pretrained(
        args.model_name_or_path,
        config=config,
        **kwargs
    )
    print('finish tfm_model')
    model = T5FineTuner(args, tfm_model, tokenizer)
    print('finish model')

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=os.path.join(args.output_dir, '{epoch:02d}-{val_loss:.2f}'),
        monitor='val_loss',
        mode='min',
        save_top_k=1
    )

    # prepare for trainer
    train_params = dict(
        default_root_dir=args.output_dir,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gpus=args.n_gpu,
        gradient_clip_val=1.0,
        max_epochs=args.num_train_epochs,
        checkpoint_callback=checkpoint_callback,
        # profiler="simple"  ##training time     
    )

    if args.debug:
        add_params = dict(
            limit_train_batches=0.1,
            limit_val_batches=0.1,
            overfit_batches=0.01,
            fast_dev_run=True
        )
        train_params.update(add_params)

    trainer = pl.Trainer(**train_params)
    trainer.fit(model)

    # save the final model
    # model.model.save_pretrained(args.output_dir)
    # tokenizer.save_pretrained(args.output_dir) 

    print("Finish training and saving the model!")

    # evaluation
    if args.do_direct_eval:
        # print("\n****** Conduct Evaluating with the last state ******")
        print("\n****** Conduct Evaluating with the best development state ******")

        ckpt_path = checkpoint_callback.best_model_path
        print("Reload the model")
        model = T5FineTuner.load_from_checkpoint(ckpt_path, hparams=args, tfm_model=tfm_model,
                                                 tokenizer=tokenizer)
        print()
        test_dataset = get_dataset(tokenizer=tokenizer, data_type='test', args=args)
        test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, num_workers=4)
        # test_loader.object_detection_faults = test_dataset.object_detection_faults
        # print(test_loader.device)

        # compute the performance scores
        start = datetime.datetime.now()
        scores = evaluate(test_loader, model, args.num_beams, args.vinvl_region_number,
                          test_dataset.object_detection_faults)
        end = datetime.datetime.now()
        diff_time = end - start
        print("evaluate time :" + str(diff_time.microseconds * 1e-6))

        # write to file
        if not args.use_visual_feats:
            log_file_path = f"./results_log/{args.dataset}_textonly.txt"
        else:
            log_file_path = f"./results_log/{args.dataset}_VL.txt"
        local_time = time.asctime(time.localtime(time.time()))

        exp_settings = f"Dataset={args.dataset}; Train bs={args.train_batch_size}, num_epochs = {args.num_train_epochs}, seed = {args.seed}, learning rate = {args.learning_rate} \n vinvl_region_number = {args.vinvl_region_number} \n num_beams = {args.num_beams}"
        exp_results = f"F1 = {scores['f1']:.4f}; Precision = {scores['precision']:.4f}; Recall = {scores['recall']:.4f}"
        log_str = f'============================================================\n'
        log_str += f"{local_time}\n{exp_settings}\n{exp_results}\n\n"

        if not os.path.exists('./results_log'):
            os.mkdir('./results_log')

        with open(log_file_path, "a+") as f:
            f.write(log_str)

if args.do_inference:
    print("\n****** Conduct inference on trained checkpoint ******")

    # initialize the T5 model from previous checkpoint
    print(f"Load trained model from {args.checkpoint}")
    print('Note that a pretrained model is required and `do_true` should be False')
    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
    # tfm_model = T5ForConditionalGeneration.from_pretrained(args.checkpoint)
    config = create_config(args)
    tfm_model = VisionT5(config)
    # model = T5FineTuner(args, tfm_model, tokenizer)
    model = T5FineTuner.load_from_checkpoint(args.checkpoint, hparams=args, tfm_model=tfm_model,
                                             tokenizer=tokenizer)
    print()
    test_dataset = get_dataset(tokenizer=tokenizer, data_type='test', args=args)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, num_workers=4)
    # test_loader.object_detection_faults = test_dataset.object_detection_faults
    # print(test_loader.device)

    # compute the performance scores
    start = datetime.datetime.now()
    scores = evaluate(test_loader, model, args.num_beams, args.vinvl_region_number,
                      test_dataset.object_detection_faults)
    end = datetime.datetime.now()
    diff_time = end - start
    print("evaluate time :" + str(diff_time.microseconds * 1e-6))

    # write to file
    if not args.use_visual_feats:
        log_file_path = f"./results_log/{args.dataset}_textonly.txt"
    else:
        log_file_path = f"./results_log/{args.dataset}_VL.txt"
    local_time = time.asctime(time.localtime(time.time()))

    exp_settings = f"Dataset={args.dataset}; Train bs={args.train_batch_size}, num_epochs = {args.num_train_epochs}, seed = {args.seed},\n"
    exp_settings += f"learning rate = {args.learning_rate}  \n"
    exp_settings += f" vinvl_region_number = {args.vinvl_region_number}"
    exp_results = f"F1 = {scores['f1']:.4f}; Precision = {scores['precision']:.4f}; Recall = {scores['recall']:.4f}"

    log_str = f'============================================================\n'
    log_str += f"{local_time}\n{exp_settings}\n{exp_results}\n\n"

    if not os.path.exists('./results_log'):
        os.mkdir('./results_log')

    with open(log_file_path, "a+") as f:
        f.write(log_str)
