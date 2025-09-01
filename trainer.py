import os

import torch
from FlagEmbedding import BGEM3FlagModel
from transformers import AutoConfig
from transformers import AutoModel
from transformers import AutoTokenizer
from transformers import Trainer
from transformers import TrainingArguments

from data_loading import load_germandpr
from data_loading import load_germanquad
from data_loading import passthrough_collator
from model_definition import ModernBertWithActivationHeadModel
from model_support import batch_encode_attached
from model_support import batch_encode_bge_m3
from model_support import batch_encode_detached


class DetachedPooledEmbedderTrainer(Trainer):
    def __init__(
        self,
        model=None,
        model_tokenizer=None,
        activation_head=None,
        bge_m3_model=None,
        bge_m3_tokenizer=None,
        query_max_length=None,
        query_pad_to=None,
        passage_max_length=None,
        passage_pad_to=None,
        batch_size=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, model=model, **kwargs)
        self.model_tokenizer = model_tokenizer
        self.activation_head = activation_head
        self.bge_m3_model = bge_m3_model
        self.bge_m3_tokenizer = bge_m3_tokenizer
        self.query_max_length = query_max_length
        self.query_pad_to = query_pad_to
        self.passage_max_length = passage_max_length
        self.passage_pad_to = passage_pad_to
        self.batch_size = batch_size

        # Freeze parent model
        for param in self.model.parameters():
            param.requires_grad = False

        # Ensure activation_head is trainable
        for param in self.activation_head.parameters():
            param.requires_grad = True

        # Make the HF Trainer aware of the activation_head
        self.model.activation_head = self.activation_head

    def create_optimizer(self):
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.activation_head.parameters(),
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
                weight_decay=self.args.weight_decay,
            )
        return self.optimizer

    def save_model(self, output_dir=None, _internal_call=False):
        super().save_model(output_dir, _internal_call=_internal_call)
        torch.save(
            self.activation_head.state_dict(),
            os.path.join(output_dir, "activation_head.pt"),
        )

    def compute_loss(
        self, model, inputs, num_items_in_batch=None, return_outputs=False
    ):
        # inputs contains raw 'query', 'passage', 'label'
        # print(inputs)
        # exit(1)
        queries = inputs["query"]
        passages = inputs["passage"]
        # labels = inputs["label"].float()

        with torch.no_grad():
            # Teacher embeddings
            query_teacher_emb = batch_encode_bge_m3(
                self.bge_m3_model,
                self.bge_m3_tokenizer,
                queries,
                batch_size=self.batch_size,
                padding="longest",
                truncation=True,
                max_length=self.query_max_length,
                pad_to=self.query_pad_to,
            )
            passage_teacher_emb = batch_encode_bge_m3(
                self.bge_m3_model,
                self.bge_m3_tokenizer,
                passages,
                batch_size=self.batch_size,
                padding="longest",
                truncation=True,
                max_length=self.passage_max_length,
                pad_to=self.passage_pad_to,
            )

            # Similarity matrix
            sim_teacher = query_teacher_emb @ passage_teacher_emb.transpose(-2, -1)

            # Z-score normalization
            sim_teacher = (sim_teacher - sim_teacher.mean()) / sim_teacher.std()

        # Student embeddings
        query_student_emb = batch_encode_detached(
            model,
            self.model_tokenizer,
            self.activation_head,
            queries,
            batch_size=self.batch_size,
            padding="longest",
            truncation=True,
            max_length=self.query_max_length,
            pad_to=self.query_pad_to,
        )
        passage_student_emb = batch_encode_detached(
            model,
            self.model_tokenizer,
            self.activation_head,
            passages,
            batch_size=self.batch_size,
            padding="longest",
            truncation=True,
            max_length=self.passage_max_length,
            pad_to=self.passage_pad_to,
        )

        # Similarity matrix
        sim_student = query_student_emb @ passage_student_emb.transpose(-2, -1)

        # Z-score normalization
        sim_student = (sim_student - sim_student.mean()) / sim_student.std()

        # MSE loss
        loss = torch.nn.functional.mse_loss(sim_student, sim_teacher)

        return (loss, sim_student) if return_outputs else loss


class AttachedPooledEmbedderTrainer(Trainer):
    def __init__(
        self,
        model=None,
        model_tokenizer=None,
        bge_m3_model=None,
        bge_m3_tokenizer=None,
        query_max_length=None,
        query_pad_to=None,
        passage_max_length=None,
        passage_pad_to=None,
        batch_size=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, model=model, **kwargs)
        self.model_tokenizer = model_tokenizer
        self.bge_m3_model = bge_m3_model
        self.bge_m3_tokenizer = bge_m3_tokenizer
        self.query_max_length = query_max_length
        self.query_pad_to = query_pad_to
        self.passage_max_length = passage_max_length
        self.passage_pad_to = passage_pad_to
        self.batch_size = batch_size

    def create_optimizer(self):
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.model.activation_head.parameters(),
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
                weight_decay=self.args.weight_decay,
            )
        return self.optimizer

    def compute_loss(
        self, model, inputs, num_items_in_batch=None, return_outputs=False
    ):
        # inputs contains raw 'query', 'passage', 'label'
        # print(inputs)
        # exit(1)
        queries = inputs["query"]
        passages = inputs["passage"]
        # labels = inputs["label"].float()

        with torch.no_grad():
            # Teacher embeddings
            query_teacher_emb = batch_encode_bge_m3(
                self.bge_m3_model,
                self.bge_m3_tokenizer,
                queries,
                batch_size=self.batch_size,
                padding="longest",
                truncation=True,
                max_length=self.query_max_length,
                pad_to=self.query_pad_to,
            )
            passage_teacher_emb = batch_encode_bge_m3(
                self.bge_m3_model,
                self.bge_m3_tokenizer,
                passages,
                batch_size=self.batch_size,
                padding="longest",
                truncation=True,
                max_length=self.passage_max_length,
                pad_to=self.passage_pad_to,
            )

            # Similarity matrix
            sim_teacher = query_teacher_emb @ passage_teacher_emb.transpose(-2, -1)

            # Z-score normalization
            sim_teacher = (sim_teacher - sim_teacher.mean()) / sim_teacher.std()

        # Student embeddings
        query_student_emb = batch_encode_attached(
            model,
            self.model_tokenizer,
            queries,
            batch_size=self.batch_size,
            padding="longest",
            truncation=True,
            max_length=self.query_max_length,
            pad_to=self.query_pad_to,
        )
        passage_student_emb = batch_encode_attached(
            model,
            self.model_tokenizer,
            passages,
            batch_size=self.batch_size,
            padding="longest",
            truncation=True,
            max_length=self.passage_max_length,
            pad_to=self.passage_pad_to,
        )

        # Similarity matrix
        sim_student = query_student_emb @ passage_student_emb.transpose(-2, -1)

        # Z-score normalization
        sim_student = (sim_student - sim_student.mean()) / sim_student.std()

        # MSE loss
        loss = torch.nn.functional.mse_loss(sim_student, sim_teacher)

        return (loss, sim_student) if return_outputs else loss


def main():
    query_max_length = 512
    query_pad_to = 16
    passage_max_length = 2048
    passage_pad_to = 64
    batch_size = 32
    dropout = 0.1

    # set default device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)

    # Load datasets
    train_dataset = (
        load_germandpr()
    )  # returns HF Dataset with 'query', 'passage', 'label'
    eval_dataset = (
        load_germanquad()
    )  # returns HF Dataset with 'query', 'passage', 'label'

    model_class = BGEM3FlagModel(
        "models/bge-m3",
        pooling_method="mean",  # "last" "cls"
        # use_fp16=True # Setting use_fp16 to True speeds up computation with a slight performance degradation
    )

    bge_m3_tokenizer = model_class.tokenizer
    bge_m3_model = model_class.model

    student_tokenizer = AutoTokenizer.from_pretrained("models/ettin-encoder-32m")
    student_model = ModernBertWithActivationHeadModel.from_pretrained(
        "models/ettin-encoder-32m"
    )

    training_args = TrainingArguments(
        output_dir="./output",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=3,
        learning_rate=1e-3,
        logging_steps=1,
        save_strategy="steps",
        save_steps=1,
        # eval_strategy="epoch",
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )

    trainer = AttachedPooledEmbedderTrainer(
        model=student_model,
        model_tokenizer=student_tokenizer,
        bge_m3_model=bge_m3_model,
        bge_m3_tokenizer=bge_m3_tokenizer,
        query_max_length=query_max_length,
        query_pad_to=query_pad_to,
        passage_max_length=passage_max_length,
        passage_pad_to=passage_pad_to,
        batch_size=batch_size,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
        data_collator=passthrough_collator,
        args=training_args,
    )

    trainer.train()


if __name__ == "__main__":
    main()
