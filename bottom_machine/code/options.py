class StrongSupervisionOptions:
    def __init__(self):
        # ----Training----
        self.epochs = 4
        self.batch_size = 8
        self.dev_batch_size = 8
        self.log_every = 50
        self.save_every = self.log_every * 5
        self.early_stopping_patience = 2
        self.gpu = 0  # this will be the primary GPU if there are > 1 GPU
        self.resume_training = False
        self.gradient_accumulation_steps = 1  # only 1 is supported

        # ----Data sizes, sequence lengths----
        self.max_seq_len = 512
        self.max_answer_length = 15
        self.num_para_chunks = 4
        self.num_sentences_per_chunk = 18
        self.max_question_len = 35

        # ----Debugging short run----
        self.debugging_short_run = False
        self.debugging_num_iterations = self.log_every * 1
        self.debugging_num_dev_iterations = None

        # ---- Evaluation only ----
        self.dev_only = False

        # ----Data location, other paths----
        self.data_pkl_path = "../data/ans_and_sf/"
        self.train_pkl_name = "preprocessed_train.pkl"
        self.dev_pkl_name = "preprocessed_dev.pkl"

        self.experiment_name = None
        self.save_path = None
        self.acc_log_file = None
        self.training_log_file = None
        self.predictions_pkl_name = "predictions.pkl"
        self.checkpoint_name = "snapshot.pt"
        self.bert_archive = "../../bert_archive/"

        # ----Network hyperparameters----
        # one of bert-base-uncased or bert-large-uncased
        self.bert_type = "bert-base-uncased"
        if self.bert_type == "bert-base-uncased":
            self.bert_hidden_size = 768
        else:
            self.bert_hidden_size = 1024

        self.num_transformer_layers = 2
        self.transformer_hidden_size = 512

        self.dropout = 0.1  # doesn't apply to BERT
        self.learning_rate = 3e-5
        self.warmup_proportion = 0.1
        self.loss_weight_span = 1.0  # make sure this is float.
        self.loss_weight_yes_no_span = 1.0  # make sure this is float.
        self.loss_weight_supporting_fact = 1.0  # make sure this is float.
        self.loss_weight_pred_sf_count = 1.0  # make sure this is float.

    def set_experiment_name(self, name):
        self.experiment_name = name
        self.save_path = "../results/" + name + "/"
        self.acc_log_file = self.save_path + "acc_log.txt"
        self.training_log_file = self.save_path + "training_log.txt"

    def set_use_small_dataset(self, log_every=50):
        self.use_small_dataset = True
        self.log_every = log_every
        self.train_pkl_name = "preprocessed_train_small.pkl"
        self.dev_pkl_name = "preprocessed_dev_small.pkl"

    def set_debugging_short_run(self):
        self.debugging_short_run = True
        self.log_every = 10
        self.debugging_num_iterations = self.log_every * 1
        self.debugging_num_dev_iterations = 1000000000
        self.set_use_small_dataset(log_every=10)

