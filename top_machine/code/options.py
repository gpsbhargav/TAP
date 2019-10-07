class SFOnlyOptions:
    def __init__(self):

        threshold = 0.2
        self.experiment_name = "Eval on supporting facts. Threshold = {}".format(
            threshold
        )

        # ----Training----
        self.epochs = 8
        self.batch_size = 16
        self.dev_batch_size = 16
        self.log_every = 100
        self.save_every = self.log_every * 5
        self.early_stopping_patience = 10
        self.gradient_accumulation_steps = 1  # only 1 is supported
        self.gpu = 0  # this will be the primary GPU if there are > 1 GPU
        self.use_multiple_gpu = False
        self.resume_training = False

        # ----Data sizes, sequence lengths----
        self.max_seq_len = 510
        self.max_answer_length = 15

        # ----Debugging short run----
        self.debugging_short_run = False
        self.debugging_num_iterations = self.log_every * 1
        self.debugging_num_dev_iterations = 10

        # ----Train on small dataset ?----
        self.use_small_dataset = False

        # ---- Evaluation only ----
        self.dev_only = False

        # ----Data location, other paths----
        self.data_pkl_path = "../data/according_to_predicted_sf/"
        if self.use_small_dataset:
            self.train_pkl_name = "preprocessed_train_small.pkl"
            self.dev_pkl_name = "preprocessed_dev_small.pkl"
        else:
            self.train_pkl_name = "preprocessed_train.pkl"
            self.dev_pkl_name = "preprocessed_dev_t_{}.pkl".format(threshold)

        self.save_path = "../results/tfef_t_{}/".format(threshold)
        self.acc_log_file = "../results/tfef_t_{}/acc_log.txt".format(threshold)
        self.predictions_pkl_name = "predictions.pkl"
        self.checkpoint_name = "snapshot.pt"
        self.bert_archive = "../../bert_archive/"

        # ----Network hyperparameters----
        self.bert_type = (
            "bert-base-uncased"
        )  # one of bert-base-uncased or bert-large-uncased
        if self.bert_type == "bert-base-uncased":
            self.bert_hidden_size = 768
        else:
            self.bert_hidden_size = 1024

        self.dropout = 0.1  # doesn't apply to BERT
        self.learning_rate = 3e-5
        self.warmup_proportion = 0.1
        self.loss_weight_span = 1.0  # make sure this is float.
        self.loss_weight_yes_no_span = 1.0  # make sure this is float.

