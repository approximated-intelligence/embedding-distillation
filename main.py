import torch
from FlagEmbedding import BGEM3FlagModel
from transformers import AutoModel
from transformers import AutoTokenizer
from transformers import TrainingArguments

from data_loading import load_germandpr
from data_loading import load_germanquad
from data_loading import make_cross_product_dataset
from data_loading import passthrough_collator
from model_support import batch_encode
from model_support import batch_encode_bge_m3
from trainer import PooledEmbedderTrainer


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

    queries = ["What is BGE M3?", "Definition of BM25"]
    passages = [
        "BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.",
        "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document",
    ]

    dataset = make_cross_product_dataset(queries, passages)
    print(dataset)
    print(dataset[0])

    # Load datasets
    train_dataset = (
        load_germandpr()
    )  # returns HF Dataset with 'query', 'passage', 'label'
    print("\n ######################   DATASET   ########################")
    print(train_dataset)
    print(train_dataset[0])
    print(train_dataset[1])
    print(train_dataset[2])
    print(train_dataset[3])
    print(train_dataset[4])

    eval_dataset = (
        load_germanquad()
    )  # returns HF Dataset with 'query', 'passage', 'label'
    print(eval_dataset)
    print(eval_dataset[0])

    model_class = BGEM3FlagModel(
        "models/bge-m3",
        pooling_method="mean",  # "last" "cls"
        # use_fp16=True,  # Setting use_fp16 to True speeds up computation with a slight performance degradation
    )

    bge_m3_tokenizer = model_class.tokenizer
    bge_m3_model = model_class.model

    ettin_tokenizer = AutoTokenizer.from_pretrained("models/ettin-encoder-32m")
    ettin_model = AutoModel.from_pretrained("models/ettin-encoder-32m")

    # get hidden dim from parent model config
    hidden_dim = ettin_model.config.hidden_size

    # activation head for learned pooling
    activation_head = torch.nn.Linear(hidden_dim, 1, bias=False).to(ettin_model.device)
    # activation_head = torch.nn.Sequential(
    #     torch.nn.Linear(hidden_dim, 2 * hidden_dim),
    #     torch.nn.ReLU(),
    #     torch.nn.Dropout(p=dropout),
    #     torch.nn.Linear(2 * hidden_dim, 1),
    # ).to(ettin_model.device)

    with torch.no_grad():
        # Teacher embeddings
        query_teacher_emb = batch_encode_bge_m3(
            bge_m3_model,
            bge_m3_tokenizer,
            queries,
            batch_size=batch_size,
            padding="longest",
            truncation=True,
            max_length=query_max_length,
            pad_to=query_pad_to,
        )
        passage_teacher_emb = batch_encode_bge_m3(
            bge_m3_model,
            bge_m3_tokenizer,
            passages,
            batch_size=batch_size,
            padding="longest",
            truncation=True,
            max_length=passage_max_length,
            pad_to=passage_pad_to,
        )

        # Similarity matrix
        sim_teacher = query_teacher_emb @ passage_teacher_emb.transpose(-2, -1)
        print(sim_teacher)

        # Z-score normalization
        sim_teacher = (sim_teacher - sim_teacher.mean()) / sim_teacher.std()
        print(sim_teacher)

        # Student embeddings
        query_student_emb = batch_encode(
            ettin_model,
            ettin_tokenizer,
            activation_head,
            queries,
            batch_size=batch_size,
            padding="longest",
            truncation=True,
            max_length=query_max_length,
            pad_to=query_pad_to,
        )
        passage_student_emb = batch_encode(
            ettin_model,
            ettin_tokenizer,
            activation_head,
            passages,
            batch_size=batch_size,
            padding="longest",
            truncation=True,
            max_length=passage_max_length,
            pad_to=passage_pad_to,
        )

        # Similarity matrix
        sim_student = query_student_emb @ passage_student_emb.transpose(-2, -1)
        print(sim_student)

        # Z-score normalization
        sim_student = (sim_student - sim_student.mean()) / sim_student.std()
        print(sim_student)

    training_args = TrainingArguments(
        output_dir="./output",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=3,
        learning_rate=1e-3,
        logging_steps=1,
        save_strategy="epoch",
        eval_strategy="epoch",
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )

    trainer = PooledEmbedderTrainer(
        model=ettin_model,
        model_tokenizer=ettin_tokenizer,
        activation_head=activation_head,
        bge_m3_model=bge_m3_model,
        bge_m3_tokenizer=bge_m3_tokenizer,
        query_max_length=query_max_length,
        query_pad_to=query_pad_to,
        passage_max_length=passage_max_length,
        passage_pad_to=passage_pad_to,
        batch_size=batch_size,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=passthrough_collator,
        args=training_args,
    )

    trainer.train()


if __name__ == "__main__":
    main()
