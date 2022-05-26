import logging
import os
import time
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import lightgbm as lgb
import lightgbm_ray as lgb_ray
import torch
from lightgbm_ray.tune import _TuneLGBMRank0Mixin
from tabulate import tabulate
from torch.utils.tensorboard import SummaryWriter

from ludwig.constants import BINARY, CATEGORY, COMBINED, LOSS, MODEL_GBM, NUMBER, TEST, TRAINING, VALIDATION
from ludwig.features.feature_utils import LudwigFeatureDict
from ludwig.globals import MODEL_WEIGHTS_FILE_NAME, TRAINING_CHECKPOINTS_DIR_PATH, TRAINING_PROGRESS_TRACKER_FILE_NAME
from ludwig.models.gbm import GBM
from ludwig.models.predictor import Predictor
from ludwig.modules.metric_modules import get_initial_validation_value
from ludwig.schema.trainer import TrainerConfig
from ludwig.trainers.base import BaseTrainer
from ludwig.trainers.registry import register_ray_trainer, register_trainer
from ludwig.utils import time_utils
from ludwig.utils.checkpoint_utils import Checkpoint, CheckpointManager
from ludwig.utils.defaults import default_random_seed
from ludwig.utils.metric_utils import get_metric_names, TrainerMetric
from ludwig.utils.trainer_utils import get_new_progress_tracker

logger = logging.getLogger(__name__)


def iter_feature_metrics(features: LudwigFeatureDict) -> Iterable[Tuple[str, str]]:
    """Helper for iterating feature names and metric names."""
    for feature_name, feature in features.items():
        for metric in feature.metric_functions:
            yield feature_name, metric


@register_trainer("lightgbm_trainer", MODEL_GBM, default=True)
class LightGBMTrainer(BaseTrainer):
    TRAIN_KEY = "training"
    VALID_KEY = "validation"
    TEST_KEY = "test"

    def __init__(
        self,
        config: TrainerConfig,
        model: GBM,
        resume: float = False,
        skip_save_model: bool = False,
        skip_save_progress: bool = False,
        skip_save_log: bool = False,
        callbacks: List = None,
        random_seed: float = default_random_seed,
        horovod: Optional[Dict] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()

        self.random_seed = random_seed
        self.model = model
        self.horovod = horovod
        self.callbacks = callbacks or []
        self.skip_save_progress = skip_save_progress
        self.skip_save_model = skip_save_model

        self.eval_batch_size = config.batch_size if config.eval_batch_size is None else config.eval_batch_size
        self._validation_field = config.validation_field
        self._validation_metric = config.validation_metric
        self.reduce_learning_rate_eval_metric = config.reduce_learning_rate_eval_metric
        self.increase_batch_size_eval_metric = config.increase_batch_size_eval_metric
        self.evaluate_training_set = config.evaluate_training_set
        try:
            base_learning_rate = float(config.learning_rate)
        except ValueError:
            # TODO (ASN): Circle back on how we want to set default placeholder value
            base_learning_rate = 0.001  # Default initial learning rate for autoML.
        self.base_learning_rate = base_learning_rate
        self.early_stop = config.early_stop
        self.boosting_type = config.boosting_type
        self.tree_learner = config.tree_learner
        self.num_boost_round = config.num_boost_round
        self.max_depth = config.max_depth
        self.num_leaves = config.num_leaves
        self.min_data_in_leaf = config.min_data_in_leaf
        self.min_sum_hessian_in_leaf = config.min_sum_hessian_in_leaf
        self.feature_fraction = config.feature_fraction
        self.bagging_fraction = config.bagging_fraction
        self.pos_bagging_fraction = config.pos_bagging_fraction
        self.neg_bagging_fraction = config.neg_bagging_fraction
        self.bagging_seed = config.bagging_seed
        self.bagging_freq = config.bagging_freq
        self.feature_fraction_bynode = config.feature_fraction_bynode
        self.feature_fraction_seed = config.feature_fraction_seed
        self.extra_trees = config.extra_trees
        self.extra_seed = config.extra_seed
        self.max_delta_step = config.max_delta_step
        self.lambda_l1 = config.lambda_l1
        self.lambda_l2 = config.lambda_l2
        self.linear_lambda = config.linear_lambda
        self.min_gain_to_split = config.min_gain_to_split
        self.drop_rate = config.drop_rate
        self.max_drop = config.max_drop
        self.skip_drop = config.skip_drop
        self.xgboost_dart_mode = config.xgboost_dart_mode
        self.uniform_drop = config.uniform_drop
        self.drop_seed = config.drop_seed
        self.top_rate = config.top_rate
        self.other_rate = config.other_rate
        self.min_data_per_group = config.min_data_per_group
        self.max_cat_threshold = config.max_cat_threshold
        self.cat_l2 = config.cat_l2
        self.cat_smooth = config.cat_smooth
        self.max_cat_to_onehot = config.max_cat_to_onehot
        self.cegb_tradeoff = config.cegb_tradeoff
        self.cegb_penalty_split = config.cegb_penalty_split
        self.path_smooth = config.path_smooth
        self.verbose = config.verbose
        self.max_bin = config.max_bin

        self.device = device
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.progress_tracker = get_new_progress_tracker(
            batch_size=-1,
            learning_rate=self.base_learning_rate,
            best_eval_metric=get_initial_validation_value(self.validation_metric),
            best_reduce_learning_rate_eval_metric=get_initial_validation_value(self.reduce_learning_rate_eval_metric),
            best_increase_batch_size_eval_metric=get_initial_validation_value(self.increase_batch_size_eval_metric),
            output_features=self.model.output_features,
        )

    def tune_batch_size(
        self,
        config: Dict[str, Any],
        training_set: "Dataset",  # noqa: F821
        random_seed: int,
        max_trials: int = 10,
        halving_limit: int = 3,
    ) -> int:
        raise NotImplementedError("Tuning batch size is not supported for LightGBM.")

    def train_online(
        self,
        dataset,
    ):
        raise NotImplementedError("Online training is not supported for LightGBM.")

    @property
    def validation_field(self):
        return self._validation_field

    @property
    def validation_metric(self):
        return self._validation_metric

    def append_metrics(self, dataset_name, results, metrics_log, tables, progress_tracker):
        epoch = progress_tracker.epoch
        steps = progress_tracker.steps
        for output_feature in self.model.output_features:
            scores = [dataset_name]

            # collect metric names based on output features metrics to
            # ensure consistent order of reporting metrics
            metric_names = self.model.output_features[output_feature].metric_functions.keys()

            for metric in metric_names:
                if metric in results[output_feature]:
                    # Some metrics may have been excepted and excluded from results.
                    score = results[output_feature][metric]
                    metrics_log[output_feature][metric].append(TrainerMetric(epoch=epoch, step=steps, value=score))
                    scores.append(score)

            tables[output_feature].append(scores)

        metrics_log[COMBINED][LOSS].append(TrainerMetric(epoch=epoch, step=steps, value=results[COMBINED][LOSS]))
        tables[COMBINED].append([dataset_name, results[COMBINED][LOSS]])

        return metrics_log, tables

    def evaluation(self, dataset, dataset_name, metrics_log, tables, batch_size, progress_tracker):
        predictor = Predictor(self.model, batch_size=batch_size, horovod=self.horovod)
        metrics, predictions = predictor.batch_evaluation(dataset, collect_predictions=False, dataset_name=dataset_name)

        self.append_metrics(dataset_name, metrics, metrics_log, tables, progress_tracker)

        return metrics_log, tables

    @classmethod
    def write_eval_summary(
        cls,
        summary_writer,
        metrics,
        step,
    ):
        if not summary_writer:
            return

        for feature_name, output_feature in metrics.items():
            for metric_name, metrics in output_feature.items():
                if metrics:
                    metric_tag = f"{feature_name}/epoch_{metric_name}"
                    metric_val = metrics[-1][-1]
                    summary_writer.add_scalar(metric_tag, metric_val, global_step=step)
        summary_writer.flush()

    def run_evaluation(
        self,
        training_set,
        validation_set,
        test_set,
        progress_tracker,
        train_summary_writer,
        validation_summary_writer,
        test_summary_writer,
        output_features,
        metrics_names,
        save_path,
        loss: torch.Tensor,
        all_losses: Dict[str, torch.Tensor],
    ) -> bool:
        """Runs evaluation over training, validation, and test sets.

        Also:
        - Prints results, saves results to the progress tracker.
        - Saves the model if the validation score is the best so far
        - If there is no validation set, the model is always saved.

        Returns whether the trainer should early stop, based on validation metrics history.
        """
        start_time = time.time()
        self.callback(lambda c: c.on_eval_start(self, progress_tracker, save_path))

        if self.is_coordinator():
            logger.info(f"\nRunning evaluation for step: {progress_tracker.steps}, epoch: {progress_tracker.epoch}")

        # ================ Eval ================
        # init tables
        tables = OrderedDict()
        for output_feature_name, output_feature in output_features.items():
            tables[output_feature_name] = [[output_feature_name] + metrics_names[output_feature_name]]
        tables[COMBINED] = [[COMBINED, LOSS]]

        # eval metrics on train
        self.eval_batch_size = max(self.eval_batch_size, progress_tracker.batch_size)
        if self.evaluate_training_set:
            self.evaluation(
                training_set, "train", progress_tracker.train_metrics, tables, self.eval_batch_size, progress_tracker
            )

            self.write_eval_summary(
                summary_writer=train_summary_writer,
                metrics=progress_tracker.train_metrics,
                step=progress_tracker.steps,
            )
        else:
            # Training set is not evaluated. Add loss to the progress tracker.
            progress_tracker.train_metrics[COMBINED][LOSS].append(
                TrainerMetric(epoch=progress_tracker.epoch, step=progress_tracker.steps, value=loss.item())
            )
            for output_feature_name, loss_tensor in all_losses.items():
                progress_tracker.train_metrics[output_feature_name][LOSS].append(
                    TrainerMetric(epoch=progress_tracker.epoch, step=progress_tracker.steps, value=loss_tensor.item())
                )
                tables[output_feature_name].append(["train", loss_tensor.item()])
            tables[COMBINED].append(["train", loss.item()])

        self.write_eval_summary(
            summary_writer=train_summary_writer,
            metrics=progress_tracker.train_metrics,
            step=progress_tracker.steps,
        )

        if validation_set is not None:
            self.callback(lambda c: c.on_validation_start(self, progress_tracker, save_path))

            # eval metrics on validation set
            self.evaluation(
                validation_set,
                "vali",
                progress_tracker.validation_metrics,
                tables,
                self.eval_batch_size,
                progress_tracker,
            )

            self.write_eval_summary(
                summary_writer=validation_summary_writer,
                metrics=progress_tracker.validation_metrics,
                step=progress_tracker.steps,
            )

            self.callback(lambda c: c.on_validation_end(self, progress_tracker, save_path))

        if test_set is not None:
            self.callback(lambda c: c.on_test_start(self, progress_tracker, save_path))

            # eval metrics on test set
            self.evaluation(
                test_set, TEST, progress_tracker.test_metrics, tables, self.eval_batch_size, progress_tracker
            )

            self.write_eval_summary(
                summary_writer=test_summary_writer,
                metrics=progress_tracker.test_metrics,
                step=progress_tracker.steps,
            )

            self.callback(lambda c: c.on_test_end(self, progress_tracker, save_path))

        elapsed_time = (time.time() - start_time) * 1000.0

        if self.is_coordinator():
            logger.debug(f"Evaluation took {time_utils.strdelta(elapsed_time)}\n")
            for output_feature, table in tables.items():
                logger.info(tabulate(table, headers="firstrow", tablefmt="fancy_grid", floatfmt=".4f"))

        # Trigger eval end callback after any model weights save for complete checkpoint
        self.callback(lambda c: c.on_eval_end(self, progress_tracker, save_path))

    def _train(
        self,
        params: Dict[str, Any],
        lgb_train: lgb.Dataset,
        eval_sets: List[lgb.Dataset],
        eval_names: List[str],
    ) -> lgb.Booster:
        """Trains a LightGBM model.

        Args:
            params: parameters for LightGBM
            lgb_train: LightGBM dataset for training
            eval_sets: LightGBM datasets for evaluation
            eval_names: names of the evaluation datasets

        Returns:
            LightGBM Booster model
        """
        gbm = lgb.train(
            params,
            lgb_train,
            num_boost_round=self.num_boost_round,
            valid_sets=eval_sets,
            valid_names=eval_names,
            feature_name=list(self.model.input_features.keys()),
            # NOTE: hummingbird does not support categorical features
            # categorical_feature=categorical_features,
            callbacks=[
                lgb.early_stopping(stopping_rounds=self.early_stop),
                lgb.log_evaluation(),
            ],
        )

        return gbm

    def train(self, training_set, validation_set=None, test_set=None, save_path="model", **kwargs):
        # TODO: construct new datasets by running encoders (for text, image)

        # TODO: only single task currently
        if len(self.model.output_features) > 1:
            raise ValueError("Only single task currently supported")

        params = self._construct_lgb_params()

        lgb_train, eval_sets, eval_names = self._construct_lgb_datasets(training_set, validation_set)

        # epoch init
        start_time = time.time()

        # Reset the metrics at the start of the next epoch
        self.model.reset_metrics()

        self.callback(lambda c: c.on_epoch_start(self, self.progress_tracker, save_path))

        gbm = self._train(params, lgb_train, eval_sets, eval_names)

        # ================ Post Training Epoch ================
        self.progress_tracker.steps = gbm.current_iteration()
        self.progress_tracker.last_improvement_steps = gbm.best_iteration
        self.callback(lambda c: c.on_epoch_end(self, self.progress_tracker, save_path))

        if self.is_coordinator():
            # ========== Save training progress ==========
            logging.debug(
                f"Epoch {self.progress_tracker.epoch} took: "
                f"{time_utils.strdelta((time.time()- start_time) * 1000.0)}."
            )
            if not self.skip_save_progress:
                self.progress_tracker.save(os.path.join(save_path, TRAINING_PROGRESS_TRACKER_FILE_NAME))

        # convert to pytorch for inference, fine tuning
        self.model.compile(gbm)
        self.model = self.model.to(self.device)

        # evaluate
        train_summary_writer = None
        validation_summary_writer = None
        test_summary_writer = None
        try:
            os.makedirs(save_path, exist_ok=True)
            tensorboard_log_dir = os.path.join(save_path, "logs")

            train_summary_writer = SummaryWriter(os.path.join(tensorboard_log_dir, TRAINING))
            if validation_set is not None and validation_set.size > 0:
                validation_summary_writer = SummaryWriter(os.path.join(tensorboard_log_dir, VALIDATION))
            if test_set is not None and test_set.size > 0:
                test_summary_writer = SummaryWriter(os.path.join(tensorboard_log_dir, TEST))

            output_features = self.model.output_features
            metrics_names = get_metric_names(output_features)

            self.run_evaluation(
                training_set,
                validation_set,
                test_set,
                self.progress_tracker,
                train_summary_writer,
                validation_summary_writer,
                test_summary_writer,
                output_features,
                metrics_names,
                save_path,
                None,
                None,
            )
        finally:
            self.callback(
                lambda c: c.on_trainer_train_teardown(self, self.progress_tracker, save_path, self.is_coordinator()),
                coordinator_only=False,
            )

            if train_summary_writer is not None:
                train_summary_writer.close()
            if validation_summary_writer is not None:
                validation_summary_writer.close()
            if test_summary_writer is not None:
                test_summary_writer.close()

        if self.is_coordinator() and not self.skip_save_model:
            self._save(save_path)

        return (
            self.model,
            self.progress_tracker.train_metrics,
            self.progress_tracker.validation_metrics,
            self.progress_tracker.test_metrics,
        )

    def _construct_lgb_params(self) -> Tuple[dict, dict]:
        output_params = {}
        for feature in self.model.output_features.values():
            if feature.type() == CATEGORY:
                output_params = {
                    "objective": "multiclass",
                    "metric": ["multi_logloss"],
                    "num_class": feature.num_classes,
                }
            elif feature.type() == BINARY:
                output_params = {
                    "objective": "binary",
                    "metric": ["binary_logloss"],
                }
            elif feature.type() == NUMBER:
                output_params = {
                    "objective": "regression",
                    "metric": ["l2", "l1"],
                }
            else:
                raise ValueError(f"Output feature must be numerical, categorical, or binary, found: {feature.type}")

        # from: https://github.com/microsoft/LightGBM/blob/master/examples/python-guide/advanced_example.py
        params = {
            "boosting_type": self.boosting_type,
            "num_leaves": self.num_leaves,
            "learning_rate": self.base_learning_rate,
            "max_depth": self.max_depth,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "pos_bagging_fraction": self.pos_bagging_fraction,
            "neg_bagging_fraction": self.neg_bagging_fraction,
            "bagging_seed": self.bagging_seed,
            "bagging_freq": self.bagging_freq,
            "feature_fraction_bynode": self.feature_fraction_bynode,
            "feature_fraction_seed": self.feature_fraction_seed,
            "extra_trees": self.extra_trees,
            "extra_seed": self.extra_seed,
            "max_delta_step": self.max_delta_step,
            "lambda_l1": self.lambda_l1,
            "lambda_l2": self.lambda_l2,
            "linear_lambda": self.linear_lambda,
            "min_gain_to_split": self.min_gain_to_split,
            "drop_rate": self.drop_rate,
            "max_drop": self.max_drop,
            "skip_drop": self.skip_drop,
            "xgboost_dart_mode": self.xgboost_dart_mode,
            "uniform_drop": self.uniform_drop,
            "drop_seed": self.drop_seed,
            "top_rate": self.top_rate,
            "other_rate": self.other_rate,
            "min_data_per_group": self.min_data_per_group,
            "max_cat_threshold": self.max_cat_threshold,
            "cat_l2": self.cat_l2,
            "cat_smooth": self.cat_smooth,
            "max_cat_to_onehot": self.max_cat_to_onehot,
            "cegb_tradeoff": self.cegb_tradeoff,
            "cegb_penalty_split": self.cegb_penalty_split,
            "path_smooth": self.path_smooth,
            "verbose": self.verbose,
            "max_bin": self.max_bin,
            "tree_learner": self.tree_learner,
            "min_data_in_leaf": self.min_data_in_leaf,
            "min_sum_hessian_in_leaf": self.min_sum_hessian_in_leaf,
            "seed": self.random_seed,
            **output_params,
        }

        return params

    def _construct_lgb_datasets(self, training_set, validation_set) -> Tuple[lgb.Dataset, List[lgb.Dataset], List[str]]:
        X_train = training_set.to_df(self.model.input_features.values())
        y_train = training_set.to_df(self.model.output_features.values())

        # create dataset for lightgbm
        # if you want to re-use data, remember to set free_raw_data=False
        lgb_train = lgb.Dataset(X_train, label=y_train)

        eval_sets = [lgb_train]
        eval_names = [LightGBMTrainer.TRAIN_KEY]
        if validation_set is not None:
            X_val = validation_set.to_df(self.model.input_features.values())
            y_val = validation_set.to_df(self.model.output_features.values())
            lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
            eval_sets.append(lgb_val)
            eval_names.append(LightGBMTrainer.VALID_KEY)
        else:
            # TODO(joppe): take X% from train set as validation set
            pass

        return lgb_train, eval_sets, eval_names

    def _save(self, save_path: str):
        os.makedirs(save_path, exist_ok=True)
        training_checkpoints_path = os.path.join(save_path, TRAINING_CHECKPOINTS_DIR_PATH)
        checkpoint = Checkpoint(model=self.model)
        checkpoint_manager = CheckpointManager(checkpoint, training_checkpoints_path, device=self.device, max_to_keep=1)
        checkpoint_manager.save(1)
        checkpoint_manager.close()

        model_weights_path = os.path.join(save_path, MODEL_WEIGHTS_FILE_NAME)
        torch.save(self.model.state_dict(), model_weights_path)

    def is_coordinator(self):
        if not self.horovod:
            return True
        return self.horovod.rank() == 0

    def callback(self, fn, coordinator_only=True):
        if not coordinator_only or self.is_coordinator():
            for callback in self.callbacks:
                fn(callback)


class LogEvalDistributed(_TuneLGBMRank0Mixin):
    def __init__(self, period: int, show_stdv: bool = True):
        self.period = period
        self.show_stdv = show_stdv

    def __call__(self, env: lgb.callback.CallbackEnv):
        if not self.is_rank_0:
            return
        if self.period > 0 and env.evaluation_result_list and (env.iteration + 1) % self.period == 0:
            result = "\t".join(
                [lgb.callback._format_eval_result(x, self.show_stdv) for x in env.evaluation_result_list]
            )
            lgb.callback._log_info(f"[{env.iteration + 1}]\t{result}")


@register_ray_trainer("lightgbm_ray_trainer", MODEL_GBM, default=True)
class LightGBMRayTrainer(LightGBMTrainer):
    def __init__(
        self,
        config: TrainerConfig,
        model: GBM,
        resume: float = False,
        skip_save_model: bool = False,
        skip_save_progress: bool = False,
        skip_save_log: bool = False,
        callbacks: List = None,
        random_seed: float = default_random_seed,
        horovod: Optional[Dict] = None,
        device: Optional[str] = None,
        trainer_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        super().__init__(
            config,
            model,
            resume,
            skip_save_model,
            skip_save_progress,
            skip_save_log,
            callbacks,
            random_seed,
            horovod,
            device,
            **kwargs,
        )

        self.ray_kwargs = trainer_kwargs or {}

    def _train(
        self,
        params: Dict[str, Any],
        lgb_train: lgb_ray.RayDMatrix,
        eval_sets: List[lgb_ray.RayDMatrix],
        eval_names: List[str],
    ) -> lgb.Booster:
        """Trains a LightGBM model using ray.

        Args:
            params: parameters for LightGBM
            lgb_train: RayDMatrix dataset for training
            eval_sets: RayDMatrix datasets for evaluation
            eval_names: names of the evaluation datasets

        Returns:
            LightGBM Booster model
        """
        gbm = lgb_ray.train(
            params,
            lgb_train,
            num_boost_round=self.num_boost_round,
            valid_sets=eval_sets,
            valid_names=eval_names,
            feature_name=list(self.model.input_features.keys()),
            # NOTE: hummingbird does not support categorical features
            # categorical_feature=categorical_features,
            callbacks=[
                lgb.early_stopping(stopping_rounds=self.early_stop),
                LogEvalDistributed(10),
            ],
            ray_params=lgb_ray.RayParams(**self.ray_kwargs),
        )

        return gbm.booster_

    def _construct_lgb_datasets(
        self,
        training_set: "RayDataset",  # noqa: F821
        validation_set: Optional["RayDataset"] = None,  # noqa: F821
        test_set: Optional["RayDataset"] = None,  # noqa: F821
    ) -> Tuple[lgb_ray.RayDMatrix, List[lgb_ray.RayDMatrix], List[str]]:
        label_col = self.model.output_features.values()[0].proc_column
        # feat_names = list(self.model.input_features.keys()) + [label_col]

        lgb_train = lgb_ray.RayDMatrix(training_set.ds, label=label_col, distributed=False)

        eval_sets = [lgb_train]
        eval_names = [LightGBMTrainer.TRAIN_KEY]
        if validation_set is not None:
            lgb_val = lgb_ray.RayDMatrix(validation_set.ds, label=label_col, distributed=False)
            eval_sets.append(lgb_val)
            eval_names.append(LightGBMTrainer.VALID_KEY)

        if test_set is not None:
            lgb_test = lgb_ray.RayDMatrix(test_set.ds, label=label_col, distributed=False)
            eval_sets.append(lgb_test)
            eval_names.append(LightGBMTrainer.TEST_KEY)

        return lgb_train, eval_sets, eval_names